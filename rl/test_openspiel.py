"""Tests for the OpenSpiel game wrapper. Run: python -m rl.test_openspiel"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import pyspiel

import rl.openspiel_game  # registers simple_buckshot
from rl.openspiel_game import _SHOOT_OPPONENT, _SHOOT_SELF  # noqa: F401


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_game_loads_with_default_params():
    g = pyspiel.load_game("simple_buckshot")
    info = g.get_type()
    _assert(info.short_name == "simple_buckshot", "short name mismatch")
    _assert(g.num_players() == 2, "expected 2 players")
    _assert(g.utility_sum() == 0.0, "expected zero-sum game")
    print("ok  game_loads_with_default_params")


def test_initial_state_is_chance_node():
    g = pyspiel.load_game("simple_buckshot")
    s = g.new_initial_state()
    _assert(s.is_chance_node(), "round 1 should begin with shell shuffle")
    outcomes = s.chance_outcomes()
    # default: 2L + 2B → C(4,2) = 6 outcomes
    _assert(len(outcomes) == 6, f"expected 6 chance outcomes, got {len(outcomes)}")
    probs = [p for _, p in outcomes]
    _assert(abs(sum(probs) - 1.0) < 1e-9, f"chance probs sum {sum(probs)} != 1")
    _assert(all(abs(p - 1 / 6) < 1e-9 for p in probs), f"chance probs not uniform: {probs}")
    print("ok  initial_state_is_chance_node")


def test_random_episode_terminates_with_zero_sum():
    g = pyspiel.load_game("simple_buckshot")
    rng = np.random.default_rng(0)
    n_games = 100
    wins = Counter()
    for _ in range(n_games):
        s = g.new_initial_state()
        steps = 0
        while not s.is_terminal() and steps < 200:
            if s.is_chance_node():
                actions, probs = zip(*s.chance_outcomes())
                a = int(rng.choice(actions, p=probs))
            else:
                a = int(rng.choice(s.legal_actions()))
            s.apply_action(a)
            steps += 1
        _assert(s.is_terminal(), "game did not terminate in 200 steps")
        ret = s.returns()
        _assert(abs(sum(ret)) < 1e-9, f"returns not zero-sum: {ret}")
        _assert(set(ret) == {1.0, -1.0}, f"unexpected returns: {ret}")
        wins["p0" if ret[0] > 0 else "p1"] += 1
    _assert(30 < wins["p0"] < 70, f"win imbalance: {wins}")
    print(f"ok  random_episode_terminates_with_zero_sum (wins={dict(wins)})")


def test_clone_preserves_state():
    g = pyspiel.load_game("simple_buckshot")
    s = g.new_initial_state()
    # Take a chance action
    s.apply_action(0)
    # Take a player action
    legal = s.legal_actions()
    s.apply_action(legal[0])
    s2 = s.clone()
    _assert(s2.current_player() == s.current_player(), "current_player diverged")
    _assert(s2.is_terminal() == s.is_terminal(), "is_terminal diverged")
    _assert(s2.legal_actions() == s.legal_actions(), "legal_actions diverged")
    _assert(
        s2.information_state_string() == s.information_state_string(),
        "info_state_string diverged",
    )
    print("ok  clone_preserves_state")


def test_blank_self_shot_on_last_shell_preserved_across_reload():
    """Regression: in the OpenSpiel wrapper, a blank self-shot that empties
    the chamber must keep the turn with the shooter AFTER the reload chance
    node resolves. Before the fix, chance always handed the turn to player 0,
    stealing player 1's keep-turn.

    We force a one-shell blank chamber via direct state mutation since the
    default shell composition (≥1 live + ≥1 blank) never naturally reaches
    a state where the LAST shell is blank without additional intermediate
    play that would be fragile to express.
    """
    g = pyspiel.load_game("simple_buckshot")
    s = g.new_initial_state()
    # Resolve the initial chance node (pick any outcome).
    s.apply_action(s.chance_outcomes()[0][0])
    # Force: player 1's turn, only a single blank shell left.
    s._shells = [False]
    s._cur_player = 1
    actor_before = s.current_player()
    s.apply_action(_SHOOT_SELF)
    _assert(s.is_chance_node(), "Emptied chamber must trigger a reload chance node")
    # Resolve reload chance; actor should resume afterwards.
    s.apply_action(s.chance_outcomes()[0][0])
    _assert(
        s.current_player() == actor_before,
        f"Blank self-shot keep-turn was stolen: expected player {actor_before}, got {s.current_player()}",
    )
    print("ok  blank_self_shot_on_last_shell_preserved_across_reload")


def test_cfr_converges_quickly():
    """Sanity check: CFR+ on the default game should drive nash_conv well
    below 0.01 within 100 iterations. If this regresses, CFR isn't
    converging — likely a bug in our game's chance/transition logic."""
    from open_spiel.python.algorithms import cfr, exploitability

    g = pyspiel.load_game("simple_buckshot")
    solver = cfr.CFRPlusSolver(g)
    for _ in range(100):
        solver.evaluate_and_update_policy()
    nash = exploitability.nash_conv(g, solver.average_policy())
    _assert(nash < 0.01, f"CFR did not converge: nash_conv={nash:.6f} after 100 iters")
    print(f"ok  cfr_converges_quickly (nash_conv={nash:.6f} after 100 iters)")


def main() -> int:
    tests = [
        test_game_loads_with_default_params,
        test_initial_state_is_chance_node,
        test_random_episode_terminates_with_zero_sum,
        test_clone_preserves_state,
        test_blank_self_shot_on_last_shell_preserved_across_reload,
        test_cfr_converges_quickly,
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
