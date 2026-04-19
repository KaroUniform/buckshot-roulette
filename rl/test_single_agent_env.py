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


def test_e8_shaping_fires_and_respects_signs():
    """Verify each E8 shaping component fires in isolation and sums correctly.
    Uses a controlled step where we know exactly what happened."""
    from rl.engine import Action, Item
    # Pool with deterministic opponent that always shoots self (so opp never
    # damages agent during our test step)
    def noop_opp(obs, mask, rng):
        # Prefer USE_SMOKE (illegal without smoke), fall back to self-shot
        if mask[int(Action.SHOOT_SELF)]:
            return int(Action.SHOOT_SELF)
        return int(np.where(mask)[0][0])
    pool = OpponentPool({"noop": noop_opp})

    env = SingleAgentBuckshotEnv(
        opponent_pool=pool,
        agent_pid=0,
        damage_bonus=0.05,
        heal_bonus=0.10,
        round_survive_bonus=0.15,
    )
    env.reset(seed=12345)
    # Set up: agent HP below max (so heal can fire), known live next, has smoke
    s = env.engine.state
    s.current_player = 0
    s.players[0].hp = 1  # room for heal
    s.players[0].max_hp = 3
    s.players[1].hp = 3
    s.players[0].inventory[:] = 0
    s.players[1].inventory[:] = 0
    s.players[0].inventory[int(Item.SMOKE)] = 1
    s.shells = [True, False]  # doesn't matter for smoke

    # Step USE_SMOKE: Δhp_me = +1, Δhp_opp = 0, no reload → only heal bonus
    obs, reward, term, trunc, info = env.step(int(Action.USE_SMOKE))
    _assert(not term, "Smoke shouldn't end episode")
    _assert(abs(reward - 0.10) < 1e-6,
            f"Heal bonus expected 0.10, got {reward:.4f}")
    print("ok  e8_shaping_heal_fires_solo")

    # Fresh env: agent shoots opponent with 1 live shell → Δhp_opp = -1, damage bonus
    env2 = SingleAgentBuckshotEnv(
        opponent_pool=pool, agent_pid=0,
        damage_bonus=0.05, heal_bonus=0.10, round_survive_bonus=0.15,
    )
    env2.reset(seed=77)
    s2 = env2.engine.state
    s2.current_player = 0
    s2.players[0].hp = 3
    s2.players[1].hp = 3
    s2.shells = [True, False, True]  # live next — shooting opp deals 1, chamber still has 2
    s2.n_reloads = 0
    obs, reward, term, trunc, info = env2.step(int(Action.SHOOT_OPPONENT))
    # After: opp.hp 3→2 (damage_bonus=0.05), no heal, no reload, opp takes turn
    # Opp will self-shoot: blank or live. Either way damage_bonus for agent
    # fires only on THIS step's Δhp_opp (over the full step window).
    # Opp may take damage from self-shot → that's NOT agent's damage_bonus;
    # but it IS reflected in delta_opp since we measure before vs after the
    # whole step. So we might credit agent for opp self-damage.
    # That's intentional per the user spec ("+damage dealt" = net damage to
    # opp during our step; the agent's shot caused the encounter).
    _assert(not term and reward >= 0.05,
            f"Damage bonus should fire ≥0.05, got {reward:.4f}")
    print("ok  e8_shaping_damage_fires_on_shot")

    # Fresh env: force a reload via shooting last shell → round_survive fires
    env3 = SingleAgentBuckshotEnv(
        opponent_pool=pool, agent_pid=0,
        damage_bonus=0.0, heal_bonus=0.0, round_survive_bonus=0.15,
    )
    env3.reset(seed=99)
    s3 = env3.engine.state
    s3.current_player = 0
    s3.players[0].hp = 3
    s3.players[1].hp = 3
    s3.shells = [False]  # single blank; self-shot keeps turn and triggers reload
    reloads_before = s3.n_reloads
    obs, reward, term, trunc, info = env3.step(int(Action.SHOOT_SELF))
    _assert(env3.engine.state.n_reloads > reloads_before,
            "Reload should fire when chamber empties")
    _assert(env3.engine.state.players[0].hp > 0,
            "Agent should be alive after blank self-shot")
    _assert(reward >= 0.15,
            f"Round-survive bonus should fire ≥0.15, got {reward:.4f}")
    print("ok  e8_shaping_round_survive_fires_on_reload")

    # Back-compat: all shaping zero → reward is strictly 0 per non-terminal step
    env4 = SingleAgentBuckshotEnv(opponent_pool=pool, agent_pid=0)
    env4.reset(seed=55)
    s4 = env4.engine.state
    s4.current_player = 0
    s4.players[0].hp = 2
    s4.players[0].max_hp = 3
    s4.players[0].inventory[:] = 0
    s4.players[0].inventory[int(Item.SMOKE)] = 1
    obs, reward, term, trunc, info = env4.step(int(Action.USE_SMOKE))
    _assert(not term and reward == 0.0,
            f"Zero-shaping reward must be exactly 0.0, got {reward}")
    print("ok  e8_shaping_zero_coefs_stays_sparse")


def main() -> int:
    tests = [
        test_obs_and_action_spaces_present,
        test_two_envs_without_seed_diverge,
        test_reset_with_opponent_first_handles_terminal_or_proceeds,
        test_random_episode_terminates_with_valid_reward,
        test_opponent_pool_sampling_distribution,
        test_policy_winrate_against_each_opponent_is_in_range,
        test_rule_based_opponents_never_pick_illegal_action,
        test_e8_shaping_fires_and_respects_signs,
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
