"""Tiny end-to-end smoke test for the PPO loop.

Runs a single update (one rollout + one PPO update epoch over a few
minibatches) to catch wiring bugs without the cost of real training.

Run: python -m rl.test_ppo
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

from rl.ppo import PPOConfig, record_terminal_returns, train


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_record_terminal_returns_counts_each_done_exactly_once():
    """Pure-function regression: a single `done==True` event must contribute
    exactly one entry per env. Earlier the PPO loop accidentally walked
    both `done[]` and `infos["final_info"]`, double-counting every episode.
    """
    buf: list = []
    # 4 envs; envs 0 and 2 terminated this step
    done = np.array([True, False, True, False])
    reward = np.array([1.0, 0.0, -1.0, 0.0])
    n = record_terminal_returns(buf, done, reward, max_size=10)
    _assert(n == 2, f"Expected 2 terminations counted, got {n}")
    _assert(buf == [1.0, -1.0], f"Buffer should hold the two terminal rewards: {buf}")

    # Calling again with no terminations should not change anything
    n2 = record_terminal_returns(buf, np.zeros(4, dtype=bool), np.zeros(4), max_size=10)
    _assert(n2 == 0 and buf == [1.0, -1.0], f"No-op call corrupted buffer: {buf}")
    print("ok  record_terminal_returns_counts_each_done_exactly_once")


def test_record_terminal_returns_respects_max_size():
    buf = [float(i) for i in range(10)]  # [0, 1, ..., 9]
    done = np.array([True])
    reward = np.array([99.0])
    record_terminal_returns(buf, done, reward, max_size=10)
    _assert(len(buf) == 10, f"Buffer should stay at max_size, got {len(buf)}")
    _assert(buf[-1] == 99.0, f"Newest value should be at the end: {buf}")
    _assert(buf[0] == 1.0, f"Oldest (0.0) should have been evicted: {buf[0]}")
    print("ok  record_terminal_returns_respects_max_size")


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


def test_ppo_termination_count_logged():
    """Smoke check that the train() loop wires record_terminal_returns into
    the metrics file. The strict per-call invariant is covered by the unit
    tests above — here we just confirm the pipeline exposes the count."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PPOConfig(
            total_timesteps=256,
            num_envs=4,
            num_steps=16,
            num_minibatches=2,
            update_epochs=2,
            snapshot_every_updates=1,
            eval_every_updates=10,
            eval_episodes=4,
            save_dir=tmp,
            run_name="smoke",
        )
        train(cfg)
        with open(os.path.join(tmp, "smoke", "metrics.jsonl")) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        last = lines[-1]
        n_term = last["rollout/n_terminations_total"]
        # Must complete at least one episode in 256 env-steps
        _assert(n_term >= 1, f"No terminations recorded: {n_term}")
        # Cap at total env-steps (terminations can never exceed steps taken)
        _assert(
            n_term <= cfg.batch_size,
            f"More terminations ({n_term}) than env-steps ({cfg.batch_size}) — impossible",
        )
    print(f"ok  ppo_termination_count_logged (n_term={n_term})")


def main() -> int:
    failures = 0
    tests = [
        test_record_terminal_returns_counts_each_done_exactly_once,
        test_record_terminal_returns_respects_max_size,
        test_ppo_single_update_runs_clean,
        test_ppo_termination_count_logged,
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
