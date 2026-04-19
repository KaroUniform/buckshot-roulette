"""Tiny end-to-end smoke test for the PPO loop.

Runs a single update (one rollout + one PPO update epoch over a few
minibatches) to catch wiring bugs without the cost of real training.

Run: python -m rl.test_ppo
"""

from __future__ import annotations

import sys
import tempfile

from rl.ppo import PPOConfig, train


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_ppo_single_update_runs_clean():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PPOConfig(
            total_timesteps=256,  # batch_size=64 → 4 updates? no: see below
            num_envs=4,
            num_steps=16,  # batch = 4 * 16 = 64
            num_minibatches=2,
            update_epochs=2,
            snapshot_every_updates=1,
            eval_every_updates=1,
            eval_episodes=8,
            save_dir=tmp,
            run_name="smoke",
        )
        # batch_size = 64; total/batch = 4 updates
        policy = train(cfg)
        # Policy should have all params with finite values
        for p in policy.parameters():
            _assert(p.isfinite().all().item(), "Non-finite parameter after training")
    print("ok  ppo_single_update_runs_clean")


def main() -> int:
    failures = 0
    for t in [test_ppo_single_update_runs_clean]:
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
