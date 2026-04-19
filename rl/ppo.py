"""CleanRL-style PPO with self-play league for Buckshot Roulette.

Single-file, readable. Uses gymnasium SyncVectorEnv for parallel rollouts.
Each env independently samples its opponent at reset() from a shared pool
that grows as we periodically snapshot the training policy.

Run: `python -m rl.ppo --total-timesteps 200000`
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.engine import NUM_ACTIONS
from rl.eval import evaluate_policy
from rl.opponents import (
    NAMED_OPPONENTS,
    OpponentPool,
    make_frozen_policy_opponent,
)
from rl.policy import ActorCritic
from rl.single_agent_env import SingleAgentBuckshotEnv


@dataclass
class PPOConfig:
    total_timesteps: int = 200_000
    num_envs: int = 32
    num_steps: int = 128  # rollout length per env
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
    hidden: int = 256
    seed: int = 1
    device: str = "cpu"
    snapshot_every_updates: int = 10
    max_pool_snapshots: int = 8
    eval_every_updates: int = 5
    eval_episodes: int = 100
    save_dir: str = "rl_runs"
    run_name: str = field(default_factory=lambda: f"ppo_{int(time.time())}")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


def record_terminal_returns(
    buffer: list, done: np.ndarray, reward: np.ndarray, max_size: int = 200
) -> int:
    """Append terminal-step rewards for each env that ended this step.

    Pure helper so the metric-collection invariant ("each completed episode
    contributes exactly once") can be unit-tested in isolation. The earlier
    PPO loop also walked `infos["final_info"]`, which double-counted because
    SyncVectorEnv populates it in lockstep with `done`.
    """
    n = 0
    for env_id in range(len(done)):
        if done[env_id]:
            buffer.append(float(reward[env_id]))
            n += 1
            if len(buffer) > max_size:
                buffer.pop(0)
    return n


def make_env_fn(pool: OpponentPool, seed: int):
    def thunk():
        env = SingleAgentBuckshotEnv(opponent_pool=pool)
        env.reset(seed=seed)
        return env

    return thunk


def train(cfg: PPOConfig) -> ActorCritic:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)
    run_dir = os.path.join(cfg.save_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Initial league: random + rule-based
    pool = OpponentPool(dict(NAMED_OPPONENTS))

    # Vectorized env
    envs = gym.vector.SyncVectorEnv(
        [make_env_fn(pool, cfg.seed + i) for i in range(cfg.num_envs)]
    )

    obs_dim = envs.single_observation_space["observation"].shape[0]
    policy = ActorCritic(obs_dim, NUM_ACTIONS, hidden=cfg.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Rollout buffers
    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=device)
    mask_buf = torch.zeros((cfg.num_steps, cfg.num_envs, NUM_ACTIONS), dtype=torch.int8, device=device)
    actions_buf = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rewards_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    dones_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    values_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

    # Reset
    obs_dict, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.from_numpy(obs_dict["observation"]).to(device)
    next_mask = torch.from_numpy(obs_dict["action_mask"]).to(device)
    next_done = torch.zeros(cfg.num_envs, device=device)

    num_updates = cfg.total_timesteps // cfg.batch_size
    global_step = 0
    ep_returns_window: list[float] = []
    n_terminations_total = 0  # for logging + regression-test on duplicate counting
    log_path = os.path.join(run_dir, "metrics.jsonl")
    log_file = open(log_path, "w")

    print(f"[{cfg.run_name}] {num_updates} updates of {cfg.batch_size} steps each "
          f"(envs={cfg.num_envs}, rollout={cfg.num_steps}) on {device}")

    try:
        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                for pg in optimizer.param_groups:
                    pg["lr"] = frac * cfg.learning_rate

            # ---- Rollout ----
            for step in range(cfg.num_steps):
                global_step += cfg.num_envs
                obs_buf[step] = next_obs
                mask_buf[step] = next_mask
                dones_buf[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = policy.get_action_and_value(next_obs, next_mask)
                    values_buf[step] = value
                actions_buf[step] = action
                logprobs_buf[step] = logprob

                obs_dict, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                done = np.logical_or(terminations, truncations)
                rewards_buf[step] = torch.from_numpy(reward.astype(np.float32)).to(device)
                next_done = torch.from_numpy(done.astype(np.float32)).to(device)
                next_obs = torch.from_numpy(obs_dict["observation"]).to(device)
                next_mask = torch.from_numpy(obs_dict["action_mask"]).to(device)

                # Episodic return = the terminal step's reward (±1 zero-sum).
                n_terminations_total += record_terminal_returns(
                    ep_returns_window, done, reward
                )

            # ---- GAE ----
            with torch.no_grad():
                _, _, _, next_value = policy.get_action_and_value(next_obs, next_mask)
                advantages = torch.zeros_like(rewards_buf)
                last_gae = 0.0
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_v = next_value
                    else:
                        next_non_terminal = 1.0 - dones_buf[t + 1]
                        next_v = values_buf[t + 1]
                    delta = rewards_buf[t] + cfg.gamma * next_v * next_non_terminal - values_buf[t]
                    last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
                    advantages[t] = last_gae
                returns = advantages + values_buf

            # ---- PPO update ----
            b_obs = obs_buf.reshape(-1, obs_dim)
            b_masks = mask_buf.reshape(-1, NUM_ACTIONS)
            b_actions = actions_buf.reshape(-1)
            b_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)

            b_inds = np.arange(cfg.batch_size)
            clipfracs: list[float] = []
            approx_kls: list[float] = []  # per-minibatch KLs across the whole update
            for epoch in range(cfg.update_epochs):
                np.random.shuffle(b_inds)
                epoch_kls: list[float] = []
                for start in range(0, cfg.batch_size, cfg.minibatch_size):
                    end = start + cfg.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, new_logprob, entropy, new_value = policy.get_action_and_value(
                        b_obs[mb_inds], b_masks[mb_inds], b_actions[mb_inds]
                    )
                    logratio = new_logprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                    epoch_kls.append(approx_kl)
                    approx_kls.append(approx_kl)

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
                    ent_loss = entropy.mean()

                    loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                # Early stop based on the EPOCH MEAN KL, not a single minibatch.
                if cfg.target_kl is not None and float(np.mean(epoch_kls)) > cfg.target_kl:
                    break

            # Report the update-level mean KL (average across all minibatches)
            approx_kl_avg = float(np.mean(approx_kls)) if approx_kls else 0.0

            # ---- League snapshot ----
            if update % cfg.snapshot_every_updates == 0:
                snap = copy.deepcopy(policy).eval()
                for p in snap.parameters():
                    p.requires_grad_(False)
                snap_name = f"snapshot_u{update}"
                pool.add(snap_name, make_frozen_policy_opponent(snap, device=cfg.device), weight=1.0)
                # Persist to disk so we can analyse training trajectory later.
                ckpts_dir = os.path.join(run_dir, "checkpoints")
                os.makedirs(ckpts_dir, exist_ok=True)
                torch.save(snap.state_dict(), os.path.join(ckpts_dir, f"{snap_name}.pt"))
                # Trim oldest snapshots beyond cap (in pool only — keep all on disk)
                snap_names = [n for n in pool.opponents if n.startswith("snapshot_")]
                while len(snap_names) > cfg.max_pool_snapshots:
                    drop = snap_names.pop(0)
                    del pool.opponents[drop]
                    del pool.weights[drop]

            # ---- Logging + eval ----
            recent_return = float(np.mean(ep_returns_window[-50:])) if ep_returns_window else 0.0
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

            if update % cfg.eval_every_updates == 0 or update == 1:
                wr = evaluate_policy(
                    policy, opponents=NAMED_OPPONENTS,
                    n_episodes=cfg.eval_episodes, seed=cfg.seed + update,
                    device=cfg.device,
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
            envs.close()
        except Exception:
            # Never let teardown raise over the primary exception
            pass
    ckpt = os.path.join(run_dir, "policy_final.pt")
    torch.save(policy.state_dict(), ckpt)
    print(f"saved final policy to {ckpt}")
    return policy


def parse_args() -> PPOConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="rl_runs")
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=100)
    a = p.parse_args()
    cfg = PPOConfig(
        total_timesteps=a.total_timesteps,
        num_envs=a.num_envs,
        num_steps=a.num_steps,
        learning_rate=a.lr,
        ent_coef=a.ent_coef,
        hidden=a.hidden,
        seed=a.seed,
        device=a.device,
        save_dir=a.save_dir,
        snapshot_every_updates=a.snapshot_every,
        eval_every_updates=a.eval_every,
        eval_episodes=a.eval_episodes,
    )
    if a.run_name:
        cfg.run_name = a.run_name
    return cfg


if __name__ == "__main__":
    train(parse_args())
