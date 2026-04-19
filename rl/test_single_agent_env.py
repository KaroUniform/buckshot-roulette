"""Tests for the single-agent gym wrapper. Run: python -m rl.test_single_agent_env"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from rl.opponents import (
    NAMED_OPPONENTS,
    OpponentPool,
    aggressive_opponent,
    conservative_opponent,
    random_opponent,
)
from rl.single_agent_env import SingleAgentBuckshotEnv


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_two_envs_without_seed_diverge():
    """Two fresh envs with no explicit seed must NOT produce identical games.
    Same regression guard as the AEC env."""
    games = []
    for _ in range(2):
        env = SingleAgentBuckshotEnv(opponent_pool=OpponentPool({"random": random_opponent}))
        env.reset()
        s = env.engine.state
        games.append(
            (
                tuple(s.shells),
                s.current_player,
                tuple(s.players[0].inventory.tolist()),
                tuple(s.players[1].inventory.tolist()),
                s.players[0].hp,
            )
        )
    _assert(games[0] != games[1], f"Two no-seed envs produced identical games: {games[0]}")
    print("ok  two_envs_without_seed_diverge")


def test_obs_and_action_spaces_present():
    env = SingleAgentBuckshotEnv()
    obs, info = env.reset(seed=0)
    _assert("observation" in obs and "action_mask" in obs, "Obs dict shape")
    _assert(obs["observation"].dtype == np.float32, "obs dtype")
    _assert(obs["action_mask"].dtype == np.int8, "mask dtype")
    _assert(env.action_space.n == 19, "action space size")
    print("ok  obs_and_action_spaces_present")


def test_reset_with_opponent_first_handles_terminal_or_proceeds():
    """If opponent goes first and somehow ends the game on their first move,
    reset() must still return a valid (obs, info) pair flagged appropriately."""
    pool = OpponentPool({"random": random_opponent})
    env = SingleAgentBuckshotEnv(opponent_pool=pool)
    # Try many seeds — at least some should have opponent going first
    found_opp_first = False
    for s in range(50):
        obs, info = env.reset(seed=s)
        if info.get("agent_pid") == 1 and not info.get("_terminated_in_reset"):
            # Engine current_player must be 1 (agent's pid) at this point
            _assert(env.engine.state.current_player == 1, "Engine should be at agent's turn")
            found_opp_first = True
        if obs["action_mask"].any() or info.get("_terminated_in_reset"):
            continue
    _assert(found_opp_first, "Never observed opponent-first start in 50 seeds")
    print("ok  reset_with_opponent_first")


def test_random_episode_terminates_with_valid_reward():
    rng = np.random.default_rng(0)
    env = SingleAgentBuckshotEnv(opponent_pool=OpponentPool({"random": random_opponent}))
    for _ in range(50):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        terminal_reward = None
        if info.get("_terminated_in_reset"):
            terminal_reward = info["_terminal_reward"]
        else:
            done = False
            while not done:
                legal = np.flatnonzero(obs["action_mask"])
                _assert(len(legal) > 0, "No legal actions mid-episode")
                a = int(rng.choice(legal))
                obs, reward, term, trunc, info = env.step(a)
                done = term or trunc
                if done:
                    terminal_reward = reward
        _assert(terminal_reward in (1.0, -1.0), f"Bad terminal reward: {terminal_reward}")
    print("ok  random_episode_terminates_with_valid_reward")


def test_opponent_pool_sampling_distribution():
    pool = OpponentPool(NAMED_OPPONENTS)
    rng = np.random.default_rng(0)
    counts = Counter(pool.sample(rng)[0] for _ in range(3000))
    for name in NAMED_OPPONENTS:
        # Equal weights → each ~1000 samples; allow wide tolerance
        _assert(700 < counts[name] < 1300, f"Sampling skew on {name}: {counts}")
    print(f"ok  opponent_pool_sampling_distribution ({dict(counts)})")


def test_policy_winrate_against_each_opponent_is_in_range():
    """Random policy should win ~50% vs random, may differ vs rule-based."""
    env_factory = lambda fn: SingleAgentBuckshotEnv(opponent_pool=OpponentPool({"x": fn}))
    rng = np.random.default_rng(0)
    for name, fn in NAMED_OPPONENTS.items():
        env = env_factory(fn)
        wins = 0
        n = 200
        for _ in range(n):
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
            tr = info.get("_terminal_reward")
            if info.get("_terminated_in_reset"):
                wins += 1 if (tr is not None and tr > 0) else 0
                continue
            done = False
            while not done:
                legal = np.flatnonzero(obs["action_mask"])
                a = int(rng.choice(legal))
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
                if done and r > 0:
                    wins += 1
        wr = wins / n
        # Basically a sanity check — shouldn't be 0% or 100%
        _assert(0.1 < wr < 0.9, f"Random vs {name} win rate suspicious: {wr:.2f}")
        print(f"     random vs {name}: {wr:.2f}")
    print("ok  policy_winrate_against_each_opponent_is_in_range")


def test_rule_based_opponents_never_pick_illegal_action():
    """Stress-test the heuristic opponents through 100 episodes each."""
    rng = np.random.default_rng(0)
    for name, fn in [("aggressive", aggressive_opponent), ("conservative", conservative_opponent)]:
        # Put the heuristic on BOTH sides to maximize state coverage
        env = SingleAgentBuckshotEnv(opponent_pool=OpponentPool({name: fn}))
        for _ in range(100):
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
            done = info.get("_terminated_in_reset", False)
            while not done:
                # Use the heuristic itself on agent side too
                obs_arr = obs["observation"]
                mask = obs["action_mask"]
                a = fn(obs_arr, mask, rng)
                _assert(mask[a] == 1, f"{name} picked illegal action {a}")
                obs, _r, term, trunc, _i = env.step(a)
                done = term or trunc
    print("ok  rule_based_opponents_never_pick_illegal_action")


def main() -> int:
    tests = [
        test_obs_and_action_spaces_present,
        test_two_envs_without_seed_diverge,
        test_reset_with_opponent_first_handles_terminal_or_proceeds,
        test_random_episode_terminates_with_valid_reward,
        test_opponent_pool_sampling_distribution,
        test_policy_winrate_against_each_opponent_is_in_range,
        test_rule_based_opponents_never_pick_illegal_action,
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
