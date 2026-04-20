"""Smoke test for the recurrent PPO loop. Mirrors test_ppo.py's shape.

Run: python -m rl.test_ppo_recurrent
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

from rl.ppo_recurrent import RecurrentPPOConfig, train


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_recurrent_ppo_smoke_runs_clean():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = RecurrentPPOConfig(
            total_timesteps=256,
            num_envs=4,
            num_steps=16,  # batch = 4 * 16 = 64
            num_minibatches=2,  # -> 2 envs per minibatch
            update_epochs=2,
            hidden=16,
            snapshot_every_updates=1,
            eval_every_updates=1,
            eval_episodes=8,
            save_dir=tmp,
            run_name="rsmoke",
        )
        policy = train(cfg)
        for p in policy.parameters():
            _assert(p.isfinite().all().item(), "non-finite weight after train()")
    print("ok  recurrent_ppo_smoke_runs_clean")


def test_recurrent_ppo_logs_terminations_and_returns():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = RecurrentPPOConfig(
            total_timesteps=256,
            num_envs=4,
            num_steps=16,
            num_minibatches=2,
            update_epochs=2,
            hidden=16,
            snapshot_every_updates=1,
            eval_every_updates=10,
            eval_episodes=4,
            save_dir=tmp,
            run_name="rsmoke2",
        )
        train(cfg)
        with open(os.path.join(tmp, "rsmoke2", "metrics.jsonl")) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        last = lines[-1]
        n_term = last["rollout/n_terminations_total"]
        _assert(n_term >= 1, f"No episodes completed: {n_term}")
        _assert(
            n_term <= cfg.total_timesteps,
            f"More terminations ({n_term}) than env-steps ({cfg.total_timesteps})",
        )
        # Basic PPO metric sanity
        _assert("approx_kl" in last, "approx_kl missing from metrics")
        _assert(np.isfinite(last["loss/policy"]), "non-finite policy loss")
        _assert(np.isfinite(last["loss/value"]), "non-finite value loss")
    print(f"ok  recurrent_ppo_logs_terminations_and_returns (n_term={n_term})")


def test_recurrent_ppo_snapshot_lands_in_pool_as_factory():
    """Self-play via recurrent snapshots: the pool entry must be a factory
    (recreatable per env) so hidden state doesn't bleed between envs."""
    from rl.ppo_recurrent import train as rtrain
    from rl.opponents import OpponentPool, NAMED_OPPONENTS

    with tempfile.TemporaryDirectory() as tmp:
        cfg = RecurrentPPOConfig(
            total_timesteps=128,
            num_envs=4,
            num_steps=16,
            num_minibatches=2,
            update_epochs=1,
            hidden=16,
            snapshot_every_updates=1,  # snap every update
            eval_every_updates=999,  # skip eval noise
            eval_episodes=2,
            save_dir=tmp,
            run_name="rsmoke3",
        )
        rtrain(cfg)
        # Inspect the snapshot directory — we can't hand back the pool, but
        # we can at least confirm snapshots were written.
        ckpts = os.path.join(tmp, "rsmoke3", "checkpoints")
        files = [f for f in os.listdir(ckpts) if f.startswith("snapshot_")]
        _assert(len(files) >= 1, f"No recurrent snapshots written: {files}")
    print(f"ok  recurrent_ppo_snapshot_lands_in_pool_as_factory (n_snap={len(files)})")


def test_recurrent_opponent_factory_isolates_hidden_state():
    """Two envs sampling the same factory must NOT share self._h. Otherwise
    hidden state written by env A corrupts env B's next action."""
    import torch
    from rl.engine import NUM_ACTIONS
    from rl.policy import RecurrentActorCritic
    from rl.opponents import make_frozen_recurrent_opponent_factory

    torch.manual_seed(0)
    policy = RecurrentActorCritic(47, NUM_ACTIONS, hidden=8)
    policy.eval()
    factory = make_frozen_recurrent_opponent_factory(policy)

    opp_a = factory()
    opp_b = factory()
    _assert(opp_a is not opp_b, "Factory must produce distinct instances")

    rng = np.random.default_rng(1)
    obs = np.zeros(47, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)

    # Mutate A's hidden by calling it once.
    _ = opp_a(obs, mask, rng)
    _assert(opp_a._h is not None, "A.h should be initialised after call")
    _assert(opp_b._h is None, "B.h must NOT be touched by A's call")

    # Now mutate B. Its hidden starts fresh, not shared with A.
    _ = opp_b(obs, mask, rng)
    # They're both initialised now, but must be SEPARATE tensors.
    _assert(opp_a._h is not opp_b._h, "A and B must hold distinct hidden tensors")
    print("ok  recurrent_opponent_factory_isolates_hidden_state")


def main() -> int:
    failures = 0
    tests = [
        test_recurrent_opponent_factory_isolates_hidden_state,
        test_recurrent_ppo_smoke_runs_clean,
        test_recurrent_ppo_logs_terminations_and_returns,
        test_recurrent_ppo_snapshot_lands_in_pool_as_factory,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            import traceback
            print(f"ERR  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
