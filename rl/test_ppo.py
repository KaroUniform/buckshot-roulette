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

    Under shaping, `reward` is a per-step delta and `running_return`
    accumulates the episode sum until termination.
    """
    buf: list = []
    # 4 envs; envs 0 and 2 terminated this step
    done = np.array([True, False, True, False])
    reward = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
    running = np.zeros(4, dtype=np.float32)
    n = record_terminal_returns(buf, done, reward, running, max_size=10)
    _assert(n == 2, f"Expected 2 terminations counted, got {n}")
    _assert(buf == [1.0, -1.0], f"Buffer should hold the two terminal returns: {buf}")
    # Terminating envs must have their running sums reset to 0.
    _assert(running[0] == 0.0 and running[2] == 0.0, f"Running sums not reset on done: {running}")

    # Calling again with no terminations should not change anything
    n2 = record_terminal_returns(
        buf,
        np.zeros(4, dtype=bool),
        np.zeros(4, dtype=np.float32),
        running,
        max_size=10,
    )
    _assert(n2 == 0 and buf == [1.0, -1.0], f"No-op call corrupted buffer: {buf}")
    print("ok  record_terminal_returns_counts_each_done_exactly_once")


def test_record_terminal_returns_accumulates_across_steps():
    """New behavior under shaping: per-step delta rewards must sum across the
    episode before the terminal return is logged."""
    buf: list = []
    running = np.zeros(1, dtype=np.float32)
    done_f = np.zeros(1, dtype=bool)
    # Three shaping steps then terminal
    record_terminal_returns(buf, done_f, np.array([0.2], dtype=np.float32), running, max_size=10)
    record_terminal_returns(buf, done_f, np.array([0.3], dtype=np.float32), running, max_size=10)
    record_terminal_returns(buf, done_f, np.array([-0.1], dtype=np.float32), running, max_size=10)
    _assert(buf == [], f"Buffer should be empty before termination: {buf}")
    record_terminal_returns(
        buf, np.array([True]), np.array([1.0], dtype=np.float32), running, max_size=10
    )
    # 0.2 + 0.3 - 0.1 + 1.0 = 1.4
    _assert(len(buf) == 1 and abs(buf[0] - 1.4) < 1e-5, f"Expected return 1.4, got {buf}")
    _assert(running[0] == 0.0, f"Running sum must reset after done: {running}")
    print("ok  record_terminal_returns_accumulates_across_steps")


def test_record_terminal_returns_respects_max_size():
    buf = [float(i) for i in range(10)]  # [0, 1, ..., 9]
    done = np.array([True])
    reward = np.array([99.0], dtype=np.float32)
    running = np.zeros(1, dtype=np.float32)
    record_terminal_returns(buf, done, reward, running, max_size=10)
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
        # Must complete at least one episode in total_timesteps env-steps
        _assert(n_term >= 1, f"No terminations recorded: {n_term}")
        # Cap at total env-steps across the WHOLE run (n_terminations_total
        # accumulates over every update, not just the last batch). Each
        # terminated env consumed at least one step, so the count can
        # never exceed total_timesteps.
        _assert(
            n_term <= cfg.total_timesteps,
            f"More terminations ({n_term}) than total env-steps ({cfg.total_timesteps}) — impossible",
        )
    print(f"ok  ppo_termination_count_logged (n_term={n_term})")


def main() -> int:
    failures = 0
    tests = [
        test_record_terminal_returns_counts_each_done_exactly_once,
        test_record_terminal_returns_accumulates_across_steps,
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
