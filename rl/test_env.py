"""Smoke tests for the PettingZoo wrapper. Run: python -m rl.test_env"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from pettingzoo.test import api_test

from rl.env import AGENTS, BuckshotAECEnv


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_pettingzoo_api_compliance():
    env = BuckshotAECEnv()
    # PettingZoo's api_test runs ~hundreds of episodes and exercises the spec.
    api_test(env, num_cycles=200, verbose_progress=False)
    print("ok  pettingzoo_api_compliance")


def test_random_play_terminates_and_balances():
    rng = np.random.default_rng(0)
    winners = Counter()
    n_games = 100
    for _ in range(n_games):
        env = BuckshotAECEnv()
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        last_winner = None
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                if reward > 0:
                    last_winner = agent
                env.step(None)
                continue
            mask = obs["action_mask"]
            legal = np.where(mask == 1)[0]
            _assert(len(legal) > 0, f"No legal actions for {agent} mid-episode")
            action = int(rng.choice(legal))
            env.step(action)
        _assert(last_winner is not None, "Game ended without a winner detected")
        winners[last_winner] += 1
    _assert(30 < winners["player_0"] < 70, f"Suspicious win imbalance: {winners}")
    print(f"ok  random_play_terminates_and_balances (winners={dict(winners)})")


def test_action_mask_is_only_set_for_current_agent():
    env = BuckshotAECEnv()
    env.reset(seed=42)
    current = env.agent_selection
    other = "player_1" if current == "player_0" else "player_0"
    obs_current = env.observe(current)
    obs_other = env.observe(other)
    _assert(obs_current["action_mask"].any(), "Current agent should have at least 1 legal action")
    _assert(not obs_other["action_mask"].any(), "Off-turn agent must have an all-zero mask")
    print("ok  action_mask_is_only_set_for_current_agent")


def test_terminal_rewards_sum_to_zero():
    """Heads-up zero-sum: winner +1, loser -1, sum = 0 every game."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        env = BuckshotAECEnv()
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        rewards = {a: 0.0 for a in AGENTS}
        for agent in env.agent_iter():
            _, r, term, trunc, _ = env.last()
            rewards[agent] += r
            if term or trunc:
                env.step(None)
                continue
            obs = env.observe(agent)
            legal = np.where(obs["action_mask"] == 1)[0]
            env.step(int(rng.choice(legal)))
        total = sum(rewards.values())
        _assert(abs(total) < 1e-6, f"Zero-sum violated: {rewards} (sum={total})")
        _assert(set(rewards.values()) == {1.0, -1.0}, f"Bad terminal rewards: {rewards}")
    print("ok  terminal_rewards_sum_to_zero")


def main() -> int:
    tests = [
        test_action_mask_is_only_set_for_current_agent,
        test_random_play_terminates_and_balances,
        test_terminal_rewards_sum_to_zero,
        test_pettingzoo_api_compliance,  # heaviest, run last
    ]
    failures = 0
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
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
