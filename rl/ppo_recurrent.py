"""Recurrent PPO (GRU actor-critic) with self-play league for Buckshot Roulette.

Sibling to ppo.py. Shares everything except the policy class and the
training-loop bookkeeping:

  * Rollout snapshots `initial_hidden` (1, num_envs, hidden) once at
    rollout start, so we can replay each env's T-length sequence during
    the PPO epochs from the correct h_0. `next_hidden` carries across
    updates (CleanRL LSTM/GRU pattern).

  * Minibatches sample ENVS (not (t, env) pairs): each minibatch is the
    full T-length rollout of a subset of envs, so BPTT flows correctly
    along the time axis within a minibatch. Without this, shuffling
    across time would sever GRU hidden-state continuity.

  * Snapshots of the training policy go into the opponent pool via the
    recurrent factory (each env-episode gets its own fresh hidden state).
    Feedforward extras still work through the old opponent wrapper.

Run: `python -m rl.ppo_recurrent --total-timesteps 200000 --hidden 128`
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.engine import NUM_ACTIONS
from rl.eval import evaluate_recurrent_policy
from rl.opponents import (
    NAMED_OPPONENTS,
    OpponentPool,
    make_frozen_policy_opponent,
    make_frozen_recurrent_opponent_factory,
)
from rl.policy import ActorCritic, RecurrentActorCritic
from rl.ppo import record_terminal_returns, make_env_fn
from rl.single_agent_env import N_OPPONENT_IDS, OPPONENT_ID_OTHER


@dataclass
class RecurrentPPOConfig:
    total_timesteps: int = 200_000
    num_envs: int = 32
    num_steps: int = 128
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden: int = 128
    seed: int = 1
    device: str = "cpu"
    snapshot_every_updates: int = 10
    max_pool_snapshots: int = 8
    eval_every_updates: int = 5
    eval_episodes: int = 100
    hp_shaping: float = 0.0
    low_hp_prob: float = 0.0
    damage_bonus: float = 0.0
    heal_bonus: float = 0.0
    round_survive_bonus: float = 0.0
    scenario_replay_prob: float = 0.0
    honest_obs: bool = False
    # Auxiliary chamber-composition loss (BEER/E12b): if True, the policy gets
    # a 2-dim regression head that predicts (n_live, n_blank) from the GRU
    # state, trained jointly via MSE with weight `aux_coef`. Used to force the
    # recurrent trunk to actually track hidden chamber state.
    aux_chamber: bool = False
    aux_coef: float = 0.05
    # E19 opponent embedding: if n_opponents > 0, the policy gets an
    # nn.Embedding(n_opponents, embed_dim) whose output is added to
    # obs_embed(obs) before the GRU. Rollout passes each env's opponent
    # id into the embedding so the trunk can condition on opponent
    # identity. Set to N_OPPONENT_IDS (5) to match the env's ID table.
    n_opponents: int = 0
    save_dir: str = "rl_runs"
    run_name: str = field(default_factory=lambda: f"ppo_recurrent_{int(time.time())}")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def envs_per_minibatch(self) -> int:
        # Minibatches sample envs (whole-T sequences), not (t, env) pairs,
        # so BPTT along the time axis stays intact within a minibatch.
        if self.num_envs % self.num_minibatches != 0:
            raise ValueError(
                f"num_envs ({self.num_envs}) must be divisible by "
                f"num_minibatches ({self.num_minibatches}) for recurrent PPO"
            )
        return self.num_envs // self.num_minibatches


def _aux_from_infos(infos: dict, num_envs: int, device: torch.device) -> torch.Tensor:
    """Extract per-env (n_live, n_blank) targets from a vector-env info dict.

    SyncVectorEnv aggregates sub-env infos into a dict of per-env arrays; on
    the occasional auto-reset step the returned info can also carry a
    `final_info`/`_final_info` entry whose shape differs, but we only consume
    n_live/n_blank which the env guarantees to emit on every step/reset path.
    Returns a tensor of shape (num_envs, 2).
    """
    n_live = np.asarray(infos.get("n_live", np.zeros(num_envs)), dtype=np.float32)
    n_blank = np.asarray(infos.get("n_blank", np.zeros(num_envs)), dtype=np.float32)
    return torch.from_numpy(np.stack([n_live, n_blank], axis=-1)).to(device)


def _opp_ids_from_infos(
    infos: dict, num_envs: int, device: torch.device, default_id: int = OPPONENT_ID_OTHER
) -> torch.Tensor:
    """Extract per-env opponent_id (int64) from a vector-env info dict.

    The env emits `opponent_id` on every reset/step path (see
    single_agent_env.py). We fall back to `default_id` only for the
    (extremely rare) case where SyncVectorEnv's info aggregation drops
    the key — e.g. if a future refactor of gymnasium changes info-dict
    merging. Returns a (num_envs,) long tensor.
    """
    ids = np.asarray(
        infos.get("opponent_id", np.full(num_envs, default_id, dtype=np.int64)),
        dtype=np.int64,
    )
    return torch.from_numpy(ids).to(device)


def train(
    cfg: RecurrentPPOConfig,
    extra_opponent_ckpts: Optional[list[str]] = None,
) -> RecurrentActorCritic:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)
    run_dir = os.path.join(cfg.save_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    pool = OpponentPool(dict(NAMED_OPPONENTS))

    if extra_opponent_ckpts:
        # Feedforward checkpoints from the old ppo.py runs still plug in as
        # league opponents. We detect FF vs recurrent by inspecting stored
        # weight keys — this keeps a mixed pool (FF baselines + future
        # recurrent snapshots) working out of the box.
        for path in extra_opponent_ckpts:
            try:
                state = torch.load(path, map_location=device, weights_only=True)
                if "gru.weight_ih_l0" in state:
                    # Recurrent extra
                    w_embed = state["obs_embed.0.weight"]
                    embed_dim, obs_dim = int(w_embed.shape[0]), int(w_embed.shape[1])
                    hidden = int(state["gru.weight_ih_l0"].shape[0] // 3)
                    aux_dim = (
                        int(state["aux_head.weight"].shape[0])
                        if "aux_head.weight" in state
                        else 0
                    )
                    n_opponents = (
                        int(state["opponent_embed.weight"].shape[0])
                        if "opponent_embed.weight" in state
                        else 0
                    )
                    net = RecurrentActorCritic(
                        obs_dim,
                        NUM_ACTIONS,
                        hidden=hidden,
                        embed=embed_dim,
                        aux_dim=aux_dim,
                        n_opponents=n_opponents,
                    ).to(device)
                    net.load_state_dict(state)
                    net.eval()
                    for p in net.parameters():
                        p.requires_grad_(False)
                    label = os.path.splitext(os.path.basename(path))[0]
                    extra_factory = make_frozen_recurrent_opponent_factory(
                        net, device=cfg.device
                    )
                    # Annotate with ckpt_path so pool.manifest() can emit a
                    # "ckpt:<path>" entry that AsyncVectorEnv workers can
                    # reload on CPU via sync_pool(manifest).
                    extra_factory.ckpt_path = os.path.abspath(path)  # type: ignore[attr-defined]
                    pool.add(f"extra:{label}", extra_factory, weight=1.0)
                    print(f"[train] added RECURRENT extra opponent {label} from {path}")
                else:
                    obs_dim = state["body.0.weight"].shape[1]
                    hidden = state["body.0.weight"].shape[0]
                    net = ActorCritic(obs_dim, NUM_ACTIONS, hidden=hidden).to(device)
                    net.load_state_dict(state)
                    net.eval()
                    for p in net.parameters():
                        p.requires_grad_(False)
                    label = os.path.splitext(os.path.basename(path))[0]
                    ff_factory = make_frozen_policy_opponent(net, device=cfg.device)
                    ff_factory.ckpt_path = os.path.abspath(path)  # type: ignore[attr-defined]
                    pool.add(f"extra:{label}", ff_factory, weight=1.0)
                    print(f"[train] added FEEDFORWARD extra opponent {label} from {path}")
            except Exception as exc:
                print(f"[train] WARNING: could not load extra opponent {path}: {exc}")

    # AsyncVectorEnv runs one worker process per env (context="spawn" so
    # CUDA in the parent doesn't poison workers). The workers get a
    # rule-only seed pool at spawn — we avoid pickling CUDA extras or
    # the threading lock through the spawn pipe — and then the parent
    # broadcasts the full pool (including recurrent-checkpoint extras)
    # via envs.call("sync_pool", pool.manifest()). shared_memory halves
    # the per-step obs-dict round-trip vs repeated pickling.
    worker_seed_pool = OpponentPool(dict(NAMED_OPPONENTS))
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env_fn(
                worker_seed_pool,
                cfg.seed + i,
                cfg.hp_shaping,
                cfg.low_hp_prob,
                cfg.damage_bonus,
                cfg.heal_bonus,
                cfg.round_survive_bonus,
                cfg.scenario_replay_prob,
                cfg.honest_obs,
            )
            for i in range(cfg.num_envs)
        ],
        context="spawn",
        shared_memory=True,
    )
    # Propagate any extras (FF/recurrent ckpts loaded above) into workers.
    # Safe to skip if no extras — the rule-only seed pool matches the
    # full pool manifest already. Always calling keeps the code uniform.
    envs.call("sync_pool", pool.manifest())

    obs_dim = envs.single_observation_space["observation"].shape[0]
    aux_dim = 2 if cfg.aux_chamber else 0
    policy = RecurrentActorCritic(
        obs_dim,
        NUM_ACTIONS,
        hidden=cfg.hidden,
        aux_dim=aux_dim,
        n_opponents=cfg.n_opponents,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Rollout buffers
    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=device)
    mask_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, NUM_ACTIONS), dtype=torch.int8, device=device
    )
    actions_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs), dtype=torch.long, device=device
    )
    logprobs_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rewards_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    dones_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    values_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    # Hidden state at rollout t=0 (shape: (1, num_envs, hidden)). Captured
    # once per rollout and used as h_0 in the PPO update to replay each
    # env's T-length sequence with BPTT. Per-step snapshots used to live
    # here too ("future debugging" in earlier revisions), but nothing
    # reads them, and the per-step write is a real GPU kernel launch.
    initial_hidden = torch.zeros((1, cfg.num_envs, cfg.hidden), device=device)
    # Aux-target buffer: ground-truth chamber composition (n_live, n_blank)
    # for the obs at each step, used to train the aux head.
    aux_target_buf = (
        torch.zeros((cfg.num_steps, cfg.num_envs, 2), device=device)
        if cfg.aux_chamber else None
    )
    # Opponent-id buffer: per-step opponent id, consumed by the embedding.
    # Allocated only when the embedding is enabled; otherwise left as None
    # so the rollout/update paths skip the extra work.
    opp_id_buf = (
        torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long, device=device)
        if cfg.n_opponents > 0 else None
    )

    obs_dict, reset_infos = envs.reset(seed=cfg.seed)
    next_obs = torch.from_numpy(obs_dict["observation"]).to(device)
    next_mask = torch.from_numpy(obs_dict["action_mask"]).to(device)
    next_done = torch.zeros(cfg.num_envs, device=device)
    next_hidden = policy.initial_hidden(cfg.num_envs, device=device)
    # aux target for the current obs: captured from env info; initially from reset
    next_aux = None
    if cfg.aux_chamber:
        next_aux = _aux_from_infos(reset_infos, cfg.num_envs, device)
    # opponent_id for the current obs: captured from env info; initially from reset
    next_opp_id = None
    if cfg.n_opponents > 0:
        next_opp_id = _opp_ids_from_infos(reset_infos, cfg.num_envs, device)

    num_updates = cfg.total_timesteps // cfg.batch_size
    global_step = 0
    ep_returns_window: list[float] = []
    ep_running_return = np.zeros(cfg.num_envs, dtype=np.float32)
    n_terminations_total = 0
    log_path = os.path.join(run_dir, "metrics.jsonl")
    log_file = open(log_path, "w")

    print(
        f"[{cfg.run_name}] RECURRENT PPO: {num_updates} updates of "
        f"{cfg.batch_size} steps each (envs={cfg.num_envs}, "
        f"rollout={cfg.num_steps}, hidden={cfg.hidden}) on {device}"
    )

    try:
        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                for pg in optimizer.param_groups:
                    pg["lr"] = frac * cfg.learning_rate

            # ---- Rollout ----
            # Snapshot h_0 once at rollout start; the PPO update replays
            # each env's full T-length sequence from this initial hidden.
            initial_hidden.copy_(next_hidden)
            for step in range(cfg.num_steps):
                global_step += cfg.num_envs
                obs_buf[step] = next_obs
                mask_buf[step] = next_mask
                dones_buf[step] = next_done
                if cfg.aux_chamber and aux_target_buf is not None:
                    aux_target_buf[step] = next_aux
                if cfg.n_opponents > 0 and opp_id_buf is not None:
                    opp_id_buf[step] = next_opp_id

                with torch.no_grad():
                    (
                        action,
                        logprob,
                        _,
                        value,
                        next_hidden,
                    ) = policy.get_action_and_value(
                        next_obs,
                        next_mask,
                        next_hidden,
                        next_done,
                        opp_ids=next_opp_id,
                    )
                    values_buf[step] = value
                actions_buf[step] = action
                logprobs_buf[step] = logprob

                obs_dict, reward, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )
                done = np.logical_or(terminations, truncations)
                rewards_buf[step] = torch.from_numpy(reward.astype(np.float32)).to(
                    device
                )
                next_done = torch.from_numpy(done.astype(np.float32)).to(device)
                next_obs = torch.from_numpy(obs_dict["observation"]).to(device)
                next_mask = torch.from_numpy(obs_dict["action_mask"]).to(device)
                if cfg.aux_chamber:
                    next_aux = _aux_from_infos(infos, cfg.num_envs, device)
                if cfg.n_opponents > 0:
                    next_opp_id = _opp_ids_from_infos(infos, cfg.num_envs, device)

                n_terminations_total += record_terminal_returns(
                    ep_returns_window, done, reward, ep_running_return
                )

            # ---- GAE ----
            with torch.no_grad():
                _, _, _, next_value, _ = policy.get_action_and_value(
                    next_obs,
                    next_mask,
                    next_hidden,
                    next_done,
                    opp_ids=next_opp_id,
                )
                advantages = torch.zeros_like(rewards_buf)
                last_gae = 0.0
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_v = next_value
                    else:
                        next_non_terminal = 1.0 - dones_buf[t + 1]
                        next_v = values_buf[t + 1]
                    delta = (
                        rewards_buf[t]
                        + cfg.gamma * next_v * next_non_terminal
                        - values_buf[t]
                    )
                    last_gae = (
                        delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
                    )
                    advantages[t] = last_gae
                returns = advantages + values_buf

            # ---- PPO update (env-minibatches; whole-T sequences) ----
            env_inds = np.arange(cfg.num_envs)
            clipfracs: list[float] = []
            approx_kls: list[float] = []
            for epoch in range(cfg.update_epochs):
                np.random.shuffle(env_inds)
                epoch_kls: list[float] = []
                for start in range(0, cfg.num_envs, cfg.envs_per_minibatch):
                    end = start + cfg.envs_per_minibatch
                    mb_envs = env_inds[start:end]
                    mb_envs_t = torch.from_numpy(mb_envs).to(device)

                    # Slices preserve the (T, B) layout so forward_sequence
                    # can replay the rollout with mid-sequence resets.
                    mb_obs = obs_buf[:, mb_envs_t]
                    mb_masks = mask_buf[:, mb_envs_t]
                    mb_actions = actions_buf[:, mb_envs_t]
                    mb_logprobs = logprobs_buf[:, mb_envs_t]
                    mb_advantages = advantages[:, mb_envs_t]
                    mb_returns = returns[:, mb_envs_t]
                    mb_values = values_buf[:, mb_envs_t]
                    mb_dones = dones_buf[:, mb_envs_t]
                    # h at t=0 for these envs (snapshot from before first step)
                    mb_h0 = initial_hidden[:, mb_envs_t].contiguous()

                    mb_opp_ids = (
                        opp_id_buf[:, mb_envs_t]
                        if cfg.n_opponents > 0 and opp_id_buf is not None
                        else None
                    )
                    if cfg.aux_chamber and aux_target_buf is not None:
                        logits_seq, values_seq, aux_pred = policy.forward_sequence(
                            mb_obs,
                            mb_h0,
                            mb_dones,
                            return_aux=True,
                            opp_ids_seq=mb_opp_ids,
                        )
                        mb_aux_target = aux_target_buf[:, mb_envs_t]
                    else:
                        logits_seq, values_seq = policy.forward_sequence(
                            mb_obs, mb_h0, mb_dones, opp_ids_seq=mb_opp_ids
                        )
                        aux_pred = None
                        mb_aux_target = None
                    masked_logits = logits_seq.masked_fill(mb_masks == 0, -1e8)
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    new_logprob = dist.log_prob(mb_actions)
                    entropy = dist.entropy()

                    logratio = new_logprob - mb_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                        )
                    epoch_kls.append(approx_kl)
                    approx_kls.append(approx_kl)

                    flat_adv = mb_advantages.reshape(-1)
                    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
                    flat_ratio = ratio.reshape(-1)

                    pg_loss1 = -flat_adv * flat_ratio
                    pg_loss2 = -flat_adv * torch.clamp(
                        flat_ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((values_seq - mb_returns) ** 2).mean()
                    ent_loss = entropy.mean()

                    loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss
                    if aux_pred is not None and mb_aux_target is not None:
                        aux_loss = ((aux_pred - mb_aux_target) ** 2).mean()
                        loss = loss + cfg.aux_coef * aux_loss
                    else:
                        aux_loss = None

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                if (
                    cfg.target_kl is not None
                    and float(np.mean(epoch_kls)) > cfg.target_kl
                ):
                    break

            approx_kl_avg = float(np.mean(approx_kls)) if approx_kls else 0.0

            # ---- League snapshot ----
            if update % cfg.snapshot_every_updates == 0:
                snap = copy.deepcopy(policy).eval()
                for p in snap.parameters():
                    p.requires_grad_(False)
                snap_name = f"snapshot_u{update}"
                ckpts_dir = os.path.join(run_dir, "checkpoints")
                os.makedirs(ckpts_dir, exist_ok=True)
                snap_ckpt_path = os.path.abspath(
                    os.path.join(ckpts_dir, f"{snap_name}.pt")
                )
                # Save FIRST so workers that reload from manifest see the
                # file on disk the moment they receive the broadcast.
                torch.save(snap.state_dict(), snap_ckpt_path)
                snap_factory = make_frozen_recurrent_opponent_factory(
                    snap, device=cfg.device
                )
                snap_factory.ckpt_path = snap_ckpt_path  # type: ignore[attr-defined]
                pool.add(snap_name, snap_factory, weight=1.0)
                snap_names = [n for n in pool.opponents if n.startswith("snapshot_")]
                while len(snap_names) > cfg.max_pool_snapshots:
                    pool.remove(snap_names.pop(0))
                # Broadcast new pool to all worker envs so their next
                # episode samples from the updated league.
                envs.call("sync_pool", pool.manifest())

            # ---- Logging + eval ----
            recent_return = (
                float(np.mean(ep_returns_window[-50:])) if ep_returns_window else 0.0
            )
            log_entry = {
                "update": update,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "loss/policy": float(pg_loss.item()),
                "loss/value": float(v_loss.item()),
                "loss/entropy": float(ent_loss.item()),
                "approx_kl": float(approx_kl_avg),
                "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "rollout/mean_return_50": recent_return,
                "rollout/n_terminations_total": n_terminations_total,
                "pool_size": len(pool),
            }
            if cfg.aux_chamber and aux_loss is not None:
                log_entry["loss/aux"] = float(aux_loss.item())

            if update % cfg.eval_every_updates == 0 or update == 1:
                wr = evaluate_recurrent_policy(
                    policy,
                    opponents=NAMED_OPPONENTS,
                    n_episodes=cfg.eval_episodes,
                    seed=cfg.seed + update,
                    device=cfg.device,
                    honest_obs=cfg.honest_obs,
                )
                for name, rate in wr.items():
                    log_entry[f"eval/winrate_vs_{name}"] = float(rate)
                print(
                    f"u{update:>4} step={global_step:>7} "
                    f"return50={recent_return:+.2f} "
                    f"vs_random={wr.get('random', 0):.2f} "
                    f"vs_aggr={wr.get('aggressive', 0):.2f} "
                    f"vs_cons={wr.get('conservative', 0):.2f} "
                    f"pool={len(pool)}"
                )

            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

    finally:
        log_file.close()
        try:
            # terminate=True forcibly SIGTERMs any worker stuck in step_wait
            # if the parent dies mid-rollout (e.g. SIGINT during training).
            envs.close(terminate=True)
        except Exception:
            pass
    ckpt = os.path.join(run_dir, "policy_final.pt")
    torch.save(policy.state_dict(), ckpt)
    print(f"saved final policy to {ckpt}")
    return policy


def parse_args() -> RecurrentPPOConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="rl_runs")
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument(
        "--extra-opponent-ckpts",
        type=str,
        default="",
        help="Comma-separated .pt paths. FF and recurrent ckpts both accepted.",
    )
    p.add_argument("--hp-shaping", type=float, default=0.0)
    p.add_argument("--low-hp-prob", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--damage-bonus", type=float, default=0.0)
    p.add_argument("--heal-bonus", type=float, default=0.0)
    p.add_argument("--round-survive-bonus", type=float, default=0.0)
    p.add_argument("--scenario-replay-prob", type=float, default=0.0)
    p.add_argument(
        "--honest-obs",
        action="store_true",
        help="Use 52-dim honest observation (no n_live/n_blank; only the "
             "initial-chamber declaration + public event counters). Required "
             "for testing whether memory+honest-info matches hack-info FF.",
    )
    p.add_argument(
        "--no-anneal-lr",
        action="store_true",
        help="Disable cosine LR anneal; see ppo.py E10 notes for why this helps "
             "escape a collapsed actor.",
    )
    p.add_argument(
        "--aux-chamber",
        action="store_true",
        help="Add a 2-dim aux head that regresses (n_live, n_blank) from the "
             "GRU state (E12b BEER representation-shaping loss).",
    )
    p.add_argument(
        "--aux-coef",
        type=float,
        default=0.05,
        help="Weight on the aux chamber-composition MSE loss (only used with "
             "--aux-chamber). 0.05 picked to be large enough to shape the "
             "trunk but small enough that policy/value remain primary.",
    )
    p.add_argument(
        "--opp-embed",
        action="store_true",
        help="Enable E19 opponent embedding (defaults n_opponents to "
             f"{N_OPPONENT_IDS} matching single_agent_env's ID table). "
             "Ignored if --n-opponents is also passed.",
    )
    p.add_argument(
        "--n-opponents",
        type=int,
        default=0,
        help="Number of opponent slots for the E19 embedding layer. 0 "
             "disables (default). Using --opp-embed sets a sensible default "
             f"({N_OPPONENT_IDS}) that matches the env's ID map. If both are "
             "passed, --n-opponents wins.",
    )
    a = p.parse_args()
    cfg = RecurrentPPOConfig(
        total_timesteps=a.total_timesteps,
        num_envs=a.num_envs,
        num_steps=a.num_steps,
        num_minibatches=a.num_minibatches,
        update_epochs=a.update_epochs,
        learning_rate=a.lr,
        ent_coef=a.ent_coef,
        hidden=a.hidden,
        seed=a.seed,
        device=a.device,
        save_dir=a.save_dir,
        snapshot_every_updates=a.snapshot_every,
        eval_every_updates=a.eval_every,
        eval_episodes=a.eval_episodes,
        hp_shaping=a.hp_shaping,
        low_hp_prob=a.low_hp_prob,
        gamma=a.gamma,
        damage_bonus=a.damage_bonus,
        heal_bonus=a.heal_bonus,
        round_survive_bonus=a.round_survive_bonus,
        scenario_replay_prob=a.scenario_replay_prob,
        honest_obs=a.honest_obs,
        anneal_lr=not a.no_anneal_lr,
        aux_chamber=a.aux_chamber,
        aux_coef=a.aux_coef,
        n_opponents=a.n_opponents if a.n_opponents > 0 else (
            N_OPPONENT_IDS if a.opp_embed else 0
        ),
    )
    if a.run_name:
        cfg.run_name = a.run_name
    extras = [s.strip() for s in a.extra_opponent_ckpts.split(",") if s.strip()]
    cfg._extra_opponent_ckpts = extras  # type: ignore
    return cfg


if __name__ == "__main__":
    _cfg = parse_args()
    _extras = getattr(_cfg, "_extra_opponent_ckpts", [])
    train(_cfg, extra_opponent_ckpts=_extras or None)
