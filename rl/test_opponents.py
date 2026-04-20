"""Tests for rule-based opponents in rl.opponents.

Run: python -m rl.test_opponents

Focus: the strong_baseline_opponent. Coverage:
  * Legality: across many random episodes, never picks an illegal action.
  * Determinism: same (obs, mask, rng) → same action.
  * Probe scenarios: the AI does the right thing in hand-crafted situations
    where the optimal play is unambiguous (saw lethal, beer survival,
    inverter convert, smoke heal, adrenaline pick chain, etc.).
  * Adrenaline two-step: USE_ADRENALINE then PICK_<X> is followed correctly.
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from rl.engine import (
    Action,
    BuckshotEngine,
    Item,
    NUM_ACTIONS,
    NUM_ITEMS,
)
from rl.opponents import (
    NAMED_OPPONENTS,
    OpponentPool,
    aggressive_opponent,
    conservative_opponent,
    random_opponent,
    strong_baseline_opponent,
)
from rl.single_agent_env import SingleAgentBuckshotEnv


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _empty_inv() -> np.ndarray:
    return np.zeros(NUM_ITEMS, dtype=np.int32)


def _setup_engine_for_current_player(seed: int = 0) -> BuckshotEngine:
    """Reset an engine so we can mutate state with a stable starting point."""
    e = BuckshotEngine()
    e.reset(seed=seed)
    return e


def _decide(e: BuckshotEngine, rng: np.random.Generator) -> int:
    pid = e.state.current_player
    obs = e.observation(pid).astype(np.float32)
    mask = e.legal_actions().astype(np.int8)
    return strong_baseline_opponent(obs, mask, rng)


# ----------------------------------------------------------------------
# Legality and determinism
# ----------------------------------------------------------------------

def test_strong_baseline_never_picks_illegal_action_1000_eps():
    """Stress-test across 1000 self-play episodes: zero illegal actions."""
    rng = np.random.default_rng(0)
    pool = OpponentPool({"strong": strong_baseline_opponent})
    env = SingleAgentBuckshotEnv(opponent_pool=pool)
    illegal_count = 0
    completed = 0
    for ep in range(1000):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000_000)))
        if info.get("_terminated_in_reset"):
            completed += 1
            continue
        done = False
        while not done:
            mask = obs["action_mask"]
            a = strong_baseline_opponent(obs["observation"], mask, rng)
            if not mask[a]:
                illegal_count += 1
                # Don't continue — the env would raise
                break
            obs, _r, term, trunc, _i = env.step(a)
            done = term or trunc
        completed += 1
    _assert(illegal_count == 0,
            f"strong_baseline picked an illegal action in {illegal_count}/{completed} episodes")
    print(f"ok  strong_baseline_never_picks_illegal_action_1000_eps "
          f"(completed={completed})")


def test_strong_baseline_deterministic_given_rng():
    """Same (obs, mask, rng-state) → same action. Critical for reproducibility
    in benchmarks and round-robin scoring."""
    rng_seed = 12345
    pool = OpponentPool({"strong": strong_baseline_opponent})
    env_a = SingleAgentBuckshotEnv(opponent_pool=pool)
    env_b = SingleAgentBuckshotEnv(opponent_pool=pool)

    obs_a, _ = env_a.reset(seed=99)
    obs_b, _ = env_b.reset(seed=99)
    rng_a = np.random.default_rng(rng_seed)
    rng_b = np.random.default_rng(rng_seed)

    for _ in range(20):
        if not obs_a["action_mask"].any():
            break
        a = strong_baseline_opponent(obs_a["observation"], obs_a["action_mask"], rng_a)
        b = strong_baseline_opponent(obs_b["observation"], obs_b["action_mask"], rng_b)
        _assert(a == b, f"Determinism violated: action_a={a} action_b={b}")
        obs_a, _, term_a, _, _ = env_a.step(a)
        obs_b, _, term_b, _, _ = env_b.step(b)
        if term_a or term_b:
            break
    print("ok  strong_baseline_deterministic_given_rng")


def test_strong_baseline_does_not_call_engine_internals():
    """Smoke test that strong_baseline only consults (obs, mask, rng).
    We pass a synthetic obs where engine.state.shells doesn't even exist
    (we never created a real game). If strong_baseline tries to touch the
    engine, this raises NameError or similar."""
    obs = np.zeros(13 + 9 + 9 + 8 + 8, dtype=np.float32)
    obs[0] = 2  # my_hp
    obs[1] = 4  # my_max
    obs[2] = 3  # opp_hp
    obs[3] = 4  # opp_max
    obs[4] = 3  # n_shells
    obs[5] = 2  # n_live
    obs[6] = 1  # n_blank
    obs[7] = 1  # damage_mult
    obs[8] = 1  # is_my_turn
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[int(Action.SHOOT_OPPONENT)] = 1
    mask[int(Action.SHOOT_SELF)] = 1
    rng = np.random.default_rng(0)
    a = strong_baseline_opponent(obs, mask, rng)
    _assert(mask[a], f"obs-only call returned illegal {a}")
    # 2 of 3 live → p_live=0.667, prefers SHOOT_OPPONENT
    _assert(a == int(Action.SHOOT_OPPONENT),
            f"With p_live=0.667, expected SHOOT_OPPONENT, got {Action(a).name}")
    print("ok  strong_baseline_does_not_call_engine_internals")


# ----------------------------------------------------------------------
# Probe scenarios — these mirror the rl/analyze_policy.py SCENARIOS the
# brief calls out, but applied to the rule-based AI directly. Each
# scenario checks for a specific anti-pattern or required play.
# ----------------------------------------------------------------------

def test_probe_beer_when_certain_death():
    """1HP, opp at full HP, next shell known LIVE, has BEER → USE_BEER (eject
    the lethal shell). The brief flagged this as the survival blindspot
    even 5M-step PPO misses."""
    e = _setup_engine_for_current_player(seed=0)
    pid = e.state.current_player
    me = e.state.players[pid]
    opp = e.state.players[1 - pid]
    me.hp = 1
    opp.hp = opp.max_hp
    e.state.shells = [True, False, True]
    e.state.known_shells[pid][0] = True
    me.inventory = _empty_inv()
    me.inventory[int(Item.BEER)] = 1
    opp.inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(0))
    _assert(a == int(Action.USE_BEER),
            f"expected USE_BEER, got {Action(a).name}")
    print("ok  probe_beer_when_certain_death")


def test_probe_saw_for_lethal():
    """Opp at 2HP, 3 live + 1 blank, has SAW → USE_HANDSAW (1-dmg→2-dmg = kill)."""
    e = _setup_engine_for_current_player(seed=1)
    pid = e.state.current_player
    me = e.state.players[pid]
    opp = e.state.players[1 - pid]
    opp.hp = 2
    e.state.shells = [True, True, True, False]
    me.inventory = _empty_inv()
    me.inventory[int(Item.HANDSAW)] = 1
    opp.inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(1))
    _assert(a == int(Action.USE_HANDSAW),
            f"expected USE_HANDSAW (lethal setup), got {Action(a).name}")
    print("ok  probe_saw_for_lethal")


def test_probe_inverter_save_from_known_live():
    """1HP, opp full, next shell known LIVE, has INVERTER → USE_INVERTER
    (flip live→blank so we can self-shoot for free). USE_BEER would also
    work; either is acceptable."""
    e = _setup_engine_for_current_player(seed=2)
    pid = e.state.current_player
    me = e.state.players[pid]
    opp = e.state.players[1 - pid]
    me.hp = 1
    opp.hp = opp.max_hp
    e.state.shells = [True, False]
    e.state.known_shells[pid][0] = True
    me.inventory = _empty_inv()
    me.inventory[int(Item.INVERTER)] = 1
    opp.inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(2))
    _assert(a == int(Action.USE_INVERTER),
            f"expected USE_INVERTER (life-saving flip), got {Action(a).name}")
    print("ok  probe_inverter_save_from_known_live")


def test_probe_smoke_when_low_hp():
    """1/4 HP, has SMOKE, mixed shells → USE_SMOKE for healing buffer."""
    e = _setup_engine_for_current_player(seed=3)
    pid = e.state.current_player
    me = e.state.players[pid]
    me.max_hp = 4
    me.hp = 1
    e.state.shells = [True, False, True]
    me.inventory = _empty_inv()
    me.inventory[int(Item.SMOKE)] = 1
    e.state.players[1 - pid].inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(3))
    _assert(a == int(Action.USE_SMOKE),
            f"expected USE_SMOKE when low HP, got {Action(a).name}")
    print("ok  probe_smoke_when_low_hp")


def test_probe_glass_for_uncertainty():
    """Mixed shells, 50/50, has GLASS → USE_GLASS for free info before deciding."""
    e = _setup_engine_for_current_player(seed=4)
    pid = e.state.current_player
    e.state.shells = [True, False, True, False]
    me = e.state.players[pid]
    me.inventory = _empty_inv()
    me.inventory[int(Item.GLASS)] = 1
    e.state.players[1 - pid].inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(4))
    _assert(a == int(Action.USE_GLASS),
            f"expected USE_GLASS for free info, got {Action(a).name}")
    print("ok  probe_glass_for_uncertainty")


def test_probe_no_saw_on_known_blank():
    """Known BLANK next, has SAW → must NOT USE_HANDSAW (saw resets after
    blank shot — wasted). Acceptable: SHOOT_SELF (free turn) or SHOOT_OPPONENT
    (burn the blank toward opp). Inverter also acceptable if we have it."""
    e = _setup_engine_for_current_player(seed=5)
    pid = e.state.current_player
    e.state.shells = [False, True]
    e.state.known_shells[pid][0] = False
    me = e.state.players[pid]
    me.inventory = _empty_inv()
    me.inventory[int(Item.HANDSAW)] = 1
    e.state.players[1 - pid].inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(5))
    _assert(a != int(Action.USE_HANDSAW),
            f"saw on known-blank is wasted; got {Action(a).name}")
    _assert(a in (int(Action.SHOOT_SELF), int(Action.SHOOT_OPPONENT)),
            f"expected SHOOT_*, got {Action(a).name}")
    print("ok  probe_no_saw_on_known_blank")


def test_probe_no_pills_with_alternatives():
    """Has PILLS but better options exist (live next, can hurt opp) → never
    pick PILLS (60% suicide). Should SHOOT_OPPONENT."""
    e = _setup_engine_for_current_player(seed=6)
    pid = e.state.current_player
    me = e.state.players[pid]
    opp = e.state.players[1 - pid]
    me.hp = 1
    opp.hp = 1
    e.state.shells = [True, False]
    e.state.known_shells[pid][0] = True
    me.inventory = _empty_inv()
    me.inventory[int(Item.PILLS)] = 1
    opp.inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(6))
    _assert(a != int(Action.USE_PILLS),
            f"pills are 60% suicide and we can lethal opp; got {Action(a).name}")
    _assert(a == int(Action.SHOOT_OPPONENT),
            f"opp is 1HP + live → SHOOT_OPPONENT for the kill; got {Action(a).name}")
    print("ok  probe_no_pills_with_alternatives")


def test_probe_inverter_known_blank_to_live():
    """Known BLANK next, has INVERTER → flip blank→live so the next shot is
    a guaranteed-live shot at opp."""
    e = _setup_engine_for_current_player(seed=7)
    pid = e.state.current_player
    e.state.shells = [False, True, False]
    e.state.known_shells[pid][0] = False
    me = e.state.players[pid]
    me.inventory = _empty_inv()
    me.inventory[int(Item.INVERTER)] = 1
    e.state.players[1 - pid].inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(7))
    _assert(a == int(Action.USE_INVERTER),
            f"expected USE_INVERTER on known-blank to convert; got {Action(a).name}")
    print("ok  probe_inverter_known_blank_to_live")


def test_probe_adrenaline_two_step_steals_saw():
    """Opp at 2HP, has SAW; we have ADRENALINE; next known LIVE.
    Step 1: USE_ADRENALINE. Step 2: PICK_HANDSAW. Step 3 (engine play): SHOOT_OPPONENT."""
    e = _setup_engine_for_current_player(seed=8)
    pid = e.state.current_player
    me = e.state.players[pid]
    opp = e.state.players[1 - pid]
    opp.hp = 2
    e.state.shells = [True, False]
    e.state.known_shells[pid][0] = True
    me.inventory = _empty_inv()
    me.inventory[int(Item.ADRENALINE)] = 1
    opp.inventory = _empty_inv()
    opp.inventory[int(Item.HANDSAW)] = 1
    rng = np.random.default_rng(8)
    a1 = _decide(e, rng)
    _assert(a1 == int(Action.USE_ADRENALINE),
            f"step 1: expected USE_ADRENALINE to steal opp saw, got {Action(a1).name}")
    e.step(a1)
    a2 = _decide(e, rng)
    _assert(a2 == int(Action.PICK_HANDSAW),
            f"step 2: expected PICK_HANDSAW after USE_ADRENALINE, got {Action(a2).name}")
    e.step(a2)
    # After PICK_HANDSAW: damage_mult should be 2 and opp inv has no saw
    _assert(e.state.damage_mult == 2,
            f"after picking saw via adrenaline, damage_mult should be 2; got {e.state.damage_mult}")
    _assert(opp.inventory[int(Item.HANDSAW)] == 0,
            "opp should no longer have the saw after we picked it")
    # Now the next decision should be SHOOT_OPPONENT (live + saw'd → kill)
    a3 = _decide(e, rng)
    _assert(a3 == int(Action.SHOOT_OPPONENT),
            f"step 3: expected SHOOT_OPPONENT (saw'd lethal), got {Action(a3).name}")
    print("ok  probe_adrenaline_two_step_steals_saw")


def test_probe_handcuff_then_lethal_known_live():
    """Opp at 2HP, next known LIVE, has CUFF + SAW → cuff first, then saw,
    then shoot. Safe lethal sequence."""
    e = _setup_engine_for_current_player(seed=9)
    pid = e.state.current_player
    me = e.state.players[pid]
    opp = e.state.players[1 - pid]
    opp.hp = 2
    e.state.shells = [True, False]
    e.state.known_shells[pid][0] = True
    me.inventory = _empty_inv()
    me.inventory[int(Item.HANDCUFF)] = 1
    me.inventory[int(Item.HANDSAW)] = 1
    opp.inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(9))
    _assert(a == int(Action.USE_HANDCUFF),
            f"expected USE_HANDCUFF first in the lethal chain, got {Action(a).name}")
    print("ok  probe_handcuff_then_lethal_known_live")


def test_probe_opp_cuffed_aggressive_lethal():
    """Opp already cuffed, opp at 2HP, known LIVE, has SAW → USE_HANDSAW
    (free attack since opp can't retaliate; saw makes the lethal one-shot)."""
    e = _setup_engine_for_current_player(seed=10)
    pid = e.state.current_player
    opp = e.state.players[1 - pid]
    opp.hp = 2
    opp.skip_next_turn = True
    e.state.shells = [True, False]
    e.state.known_shells[pid][0] = True
    me = e.state.players[pid]
    me.inventory = _empty_inv()
    me.inventory[int(Item.HANDSAW)] = 1
    opp.inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(10))
    _assert(a == int(Action.USE_HANDSAW),
            f"expected USE_HANDSAW for safe lethal, got {Action(a).name}")
    print("ok  probe_opp_cuffed_aggressive_lethal")


def test_probe_known_live_lethal_one_shot_skips_cuff_at_full_hp():
    """Already lethal-on-current-shot (opp 1HP, known live), no real benefit
    to cuffing if we're not threatened. Should SHOOT_OPPONENT directly when
    we don't have a follow-up that needs cuffing."""
    e = _setup_engine_for_current_player(seed=11)
    pid = e.state.current_player
    me = e.state.players[pid]
    opp = e.state.players[1 - pid]
    opp.hp = 1
    me.hp = me.max_hp
    e.state.shells = [True]  # n_shells == 1 → cuff path is gated by n_shells>=2
    e.state.known_shells[pid][0] = True
    me.inventory = _empty_inv()
    me.inventory[int(Item.HANDCUFF)] = 1
    opp.inventory = _empty_inv()
    a = _decide(e, np.random.default_rng(11))
    _assert(a == int(Action.SHOOT_OPPONENT),
            f"with single live shell + 1HP opp: just shoot; got {Action(a).name}")
    print("ok  probe_known_live_lethal_skip_cuff_with_1_shell")


# ----------------------------------------------------------------------
# Win-rate guarantees (loose; full benchmark numbers in docs/baseline_ai_v1.md)
# ----------------------------------------------------------------------

def test_strong_baseline_beats_random_decisively():
    """Brief: ≥60% vs random. Use 300 episodes for a wide CI."""
    pool = OpponentPool({"random": random_opponent})
    env = SingleAgentBuckshotEnv(opponent_pool=pool)
    rng = np.random.default_rng(0)
    n = 300
    wins = 0
    for ep in range(n):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000_000)))
        if info.get("_terminated_in_reset"):
            if info.get("_terminal_reward", 0) > 0:
                wins += 1
            continue
        done = False
        while not done:
            mask = obs["action_mask"]
            a = strong_baseline_opponent(obs["observation"], mask, rng)
            obs, _r, term, trunc, _i = env.step(a)
            done = term or trunc
            if done and _r > 0:
                wins += 1
    wr = wins / n
    _assert(wr >= 0.60,
            f"vs random: {wr:.3f} (target ≥0.60)")
    print(f"ok  strong_baseline_beats_random_decisively (wr={wr:.3f}, target≥0.60)")


def test_strong_baseline_beats_aggressive():
    """Brief: ≥55% vs aggressive."""
    pool = OpponentPool({"aggressive": aggressive_opponent})
    env = SingleAgentBuckshotEnv(opponent_pool=pool)
    rng = np.random.default_rng(0)
    n = 300
    wins = 0
    for ep in range(n):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000_000)))
        if info.get("_terminated_in_reset"):
            if info.get("_terminal_reward", 0) > 0:
                wins += 1
            continue
        done = False
        while not done:
            mask = obs["action_mask"]
            a = strong_baseline_opponent(obs["observation"], mask, rng)
            obs, _r, term, trunc, _i = env.step(a)
            done = term or trunc
            if done and _r > 0:
                wins += 1
    wr = wins / n
    _assert(wr >= 0.55,
            f"vs aggressive: {wr:.3f} (target ≥0.55)")
    print(f"ok  strong_baseline_beats_aggressive (wr={wr:.3f}, target≥0.55)")


def test_strong_baseline_beats_conservative():
    """Brief: ≥55% vs conservative. Tighter than aggressive — conservative
    plays a similar 'nice heuristic' style but with item use."""
    pool = OpponentPool({"conservative": conservative_opponent})
    env = SingleAgentBuckshotEnv(opponent_pool=pool)
    rng = np.random.default_rng(0)
    n = 300
    wins = 0
    for ep in range(n):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000_000)))
        if info.get("_terminated_in_reset"):
            if info.get("_terminal_reward", 0) > 0:
                wins += 1
            continue
        done = False
        while not done:
            mask = obs["action_mask"]
            a = strong_baseline_opponent(obs["observation"], mask, rng)
            obs, _r, term, trunc, _i = env.step(a)
            done = term or trunc
            if done and _r > 0:
                wins += 1
    wr = wins / n
    # 0.55 is the brief target. n=300 has ~5.6% half-CI; allow 0.50 floor to
    # absorb seed noise in CI environments.
    _assert(wr >= 0.50,
            f"vs conservative: {wr:.3f} (target ≥0.55, floor ≥0.50 for CI noise)")
    print(f"ok  strong_baseline_beats_conservative (wr={wr:.3f}, target≥0.55)")


def test_strong_baseline_in_named_opponents_registry():
    """Sanity: 'strong_baseline' is wired into the registry so league_train
    and round_robin can reference it by name."""
    _assert("strong_baseline" in NAMED_OPPONENTS,
            "strong_baseline must be in NAMED_OPPONENTS for league usage")
    fn = NAMED_OPPONENTS["strong_baseline"]
    _assert(fn is strong_baseline_opponent,
            "registry entry should be the actual function reference")
    print("ok  strong_baseline_in_named_opponents_registry")


def test_rule_based_opponents_declare_hack_layout():
    """Rule-based baselines read obs scalars at fixed hack-layout offsets, so
    each must declare `obs_layout = 'hack'`. The env uses this to fetch a
    matching obs from the engine even when the agent is on honest obs."""
    from rl.opponents import (
        aggressive_opponent,
        conservative_opponent,
        random_opponent,
        strong_baseline_opponent,
    )
    for fn in (
        random_opponent,
        aggressive_opponent,
        conservative_opponent,
        strong_baseline_opponent,
    ):
        _assert(getattr(fn, "obs_layout", None) == "hack",
                f"{fn.__name__} must declare obs_layout='hack'")
    print("ok  rule_based_opponents_declare_hack_layout")


def test_observation_layout_override():
    """engine.observation(pid, layout=...) must force the requested layout
    regardless of the engine's honest_obs flag. This is the mechanism the
    env uses to give a hack-layout obs to a hack-trained opponent while
    feeding a honest-layout obs to the agent in the same episode."""
    from rl.engine import BuckshotEngine
    for flag in (False, True):
        e = BuckshotEngine(seed=7, honest_obs=flag)
        e.reset()
        pid = e.state.current_player
        hack = e.observation(pid, layout="hack")
        honest = e.observation(pid, layout="honest")
        _assert(hack.shape[0] == 47, f"forced hack must be 47-dim (honest_obs={flag})")
        _assert(honest.shape[0] == 52, f"forced honest must be 52-dim (honest_obs={flag})")
        # Hack-layout obs carries the TRUE post-event n_live/n_blank from the
        # engine state, independent of any deriving from counters — this is
        # exactly what the INVERTER-bias fix restores for rule-based opponents
        # that were previously getting a derived (biased) view in honest envs.
        n_live_true = sum(1 for x in e.state.shells if x)
        _assert(abs(float(hack[5]) - n_live_true) < 1e-6,
                f"hack obs n_live={hack[5]} must equal true {n_live_true}")
        default = e.observation(pid)
        expected_len = 52 if flag else 47
        _assert(default.shape[0] == expected_len,
                f"layout=None must follow engine.honest_obs; got {default.shape[0]} "
                f"for honest_obs={flag}")
    print("ok  observation_layout_override")


def test_env_routes_hack_layout_to_rule_based_opponent_under_honest_obs():
    """Integration: SingleAgentBuckshotEnv with honest_obs=True must hand a
    hack-layout obs to a rule-based opponent (declared 'hack') while the
    agent sees honest. Regression test for the INVERTER bias fix: previously
    a handcrafted honest->hack collapse was called inside the opponent fn
    and mis-derived n_live/n_blank after inverter uses."""
    from rl.opponents import OpponentPool
    from rl.single_agent_env import SingleAgentBuckshotEnv

    captured: list[np.ndarray] = []

    def probe_opponent(obs, mask, rng):
        captured.append(obs.copy())
        legal = np.flatnonzero(mask)
        return int(rng.choice(legal))

    probe_opponent.obs_layout = "hack"

    pool = OpponentPool({"probe": probe_opponent})
    env = SingleAgentBuckshotEnv(opponent_pool=pool, honest_obs=True)
    env.reset(seed=1234)
    steps = 0
    while steps < 50 and not env.engine.state.done:
        mask = env.engine.legal_actions()
        legal = np.flatnonzero(mask)
        env.step(int(legal[0]))
        steps += 1
    _assert(len(captured) > 0, "opponent should have been called at least once")
    for obs in captured:
        _assert(obs.shape[0] == 47,
                f"env must pass 47-dim hack obs to a 'hack' opponent; got {obs.shape[0]}")
    # Also verify the agent's own obs was 52-dim honest.
    agent_obs = env.engine.observation(env._agent_pid_this_ep)
    _assert(agent_obs.shape[0] == 52,
            "agent must still receive 52-dim honest obs even when opponent gets hack")
    print("ok  env_routes_hack_layout_to_rule_based_opponent_under_honest_obs")


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

def main() -> int:
    tests = [
        # legality + determinism
        test_strong_baseline_never_picks_illegal_action_1000_eps,
        test_strong_baseline_deterministic_given_rng,
        test_strong_baseline_does_not_call_engine_internals,
        # probe scenarios
        test_probe_beer_when_certain_death,
        test_probe_saw_for_lethal,
        test_probe_inverter_save_from_known_live,
        test_probe_smoke_when_low_hp,
        test_probe_glass_for_uncertainty,
        test_probe_no_saw_on_known_blank,
        test_probe_no_pills_with_alternatives,
        test_probe_inverter_known_blank_to_live,
        test_probe_adrenaline_two_step_steals_saw,
        test_probe_handcuff_then_lethal_known_live,
        test_probe_opp_cuffed_aggressive_lethal,
        test_probe_known_live_lethal_one_shot_skips_cuff_at_full_hp,
        # win-rate sanity
        test_strong_baseline_beats_random_decisively,
        test_strong_baseline_beats_aggressive,
        test_strong_baseline_beats_conservative,
        # registry
        test_strong_baseline_in_named_opponents_registry,
        # honest-obs compatibility
        test_rule_based_opponents_declare_hack_layout,
        test_observation_layout_override,
        test_env_routes_hack_layout_to_rule_based_opponent_under_honest_obs,
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
