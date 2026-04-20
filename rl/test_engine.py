"""Smoke tests for BuckshotEngine. Run with: python -m rl.test_engine"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from rl.engine import (
    Action,
    BuckshotEngine,
    Item,
    NUM_ACTIONS,
)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_reset_is_deterministic_with_seed():
    e1 = BuckshotEngine(seed=42)
    s1 = e1.reset()
    e2 = BuckshotEngine(seed=42)
    s2 = e2.reset()
    _assert(s1.players[0].hp == s2.players[0].hp, "HP mismatch under same seed")
    _assert(s1.shells == s2.shells, "Shell sequence mismatch under same seed")
    _assert(s1.current_player == s2.current_player, "Starting player mismatch")
    _assert(np.array_equal(s1.players[0].inventory, s2.players[0].inventory), "Inv mismatch")
    print("ok  reset_is_deterministic_with_seed")


def test_observation_shape_is_stable():
    e = BuckshotEngine(seed=1)
    e.reset()
    o0 = e.observation(0)
    o1 = e.observation(1)
    _assert(o0.shape == o1.shape, "Per-player obs shapes differ")
    _assert(o0.dtype == np.float32, "obs dtype is not float32")
    # After a few moves the shape must remain identical
    for _ in range(20):
        if e.state.done:
            break
        legal = np.where(e.legal_actions())[0]
        if len(legal) == 0:
            break
        a = int(np.random.default_rng(0).choice(legal))
        e.step(a)
        if e.state.done:
            break
        _assert(e.observation(0).shape == o0.shape, "obs shape changed mid-episode")
    print("ok  observation_shape_is_stable")


def test_action_mask_excludes_unowned_items():
    e = BuckshotEngine(seed=7)
    e.reset()
    # Force-clear inventories so every USE_<X> must be illegal
    for p in e.state.players:
        p.inventory[:] = 0
    mask = e.legal_actions()
    for a in range(int(Action.USE_HANDSAW), int(Action.USE_INVERTER) + 1):
        _assert(not mask[a], f"{Action(a).name} should be illegal with empty inventory")
    # Shoot actions should still be legal (shells loaded)
    _assert(mask[int(Action.SHOOT_OPPONENT)], "SHOOT_OPPONENT should be legal")
    _assert(mask[int(Action.SHOOT_SELF)], "SHOOT_SELF should be legal")
    print("ok  action_mask_excludes_unowned_items")


def test_smoke_caps_at_max_hp():
    e = BuckshotEngine(seed=3)
    e.reset()
    p = e.state.players[e.state.current_player]
    p.hp = p.max_hp  # already full
    p.inventory[int(Item.SMOKE)] = 1
    _assert(not e.legal_actions()[int(Action.USE_SMOKE)], "SMOKE at full HP should be illegal")
    print("ok  smoke_caps_at_max_hp")


def test_item_usable_respects_user_and_target_args():
    """_item_usable(item, user, target) must key SMOKE off `user` and HANDCUFF
    off `target` regardless of whose turn it is. Guards against the bug-bot
    issue where SMOKE silently ignored `target` and used `current_player`."""
    e = BuckshotEngine(seed=7)
    e.reset()
    p0 = e.state.players[0]
    p1 = e.state.players[1]
    # Force a known configuration: p0 is full HP, p1 is injured; cuff p0, not p1.
    p0.hp = p0.max_hp
    p1.hp = max(1, p1.max_hp - 1)
    p0.skip_next_turn = True
    p1.skip_next_turn = False
    # SMOKE: usability must depend on `user`, not current_player.
    _assert(not e._item_usable(Item.SMOKE, user=p0, target=p1),
            "SMOKE should be unusable when user is at full HP")
    _assert(e._item_usable(Item.SMOKE, user=p1, target=p0),
            "SMOKE should be usable when user is below max HP")
    # HANDCUFF: usability must depend on `target`, not current_player.
    _assert(not e._item_usable(Item.HANDCUFF, user=p1, target=p0),
            "HANDCUFF should be illegal when target is already cuffed")
    _assert(e._item_usable(Item.HANDCUFF, user=p0, target=p1),
            "HANDCUFF should be legal when target is not cuffed")
    print("ok  item_usable_respects_user_and_target_args")


def test_handsaw_doubles_damage_then_resets():
    e = BuckshotEngine(seed=5)
    e.reset()
    # Force shells: live first, then blank — and give current player a handsaw
    e.state.shells = [True, False]
    p = e.state.players[e.state.current_player]
    opp = e.state.players[1 - e.state.current_player]
    p.inventory[int(Item.HANDSAW)] = 1
    opp_hp_before = opp.hp
    e.step(int(Action.USE_HANDSAW))
    _assert(e.state.damage_mult == 2, "Handsaw should set damage_mult to 2")
    e.step(int(Action.SHOOT_OPPONENT))
    expected_dmg = 2
    _assert(opp.hp == opp_hp_before - expected_dmg, f"Expected -{expected_dmg} HP from saw shot")
    _assert(e.state.damage_mult == 1, "damage_mult should reset after shot")
    print("ok  handsaw_doubles_damage_then_resets")


def test_blank_self_shot_keeps_turn():
    e = BuckshotEngine(seed=11)
    e.reset()
    # Force a blank as the next shell, multiple rounds remaining
    e.state.shells = [False, True, False]
    starter = e.state.current_player
    e.step(int(Action.SHOOT_SELF))
    _assert(e.state.current_player == starter, "Blank self-shot must keep turn")
    _assert(e.state.players[starter].hp > 0, "Should not lose HP on blank")
    print("ok  blank_self_shot_keeps_turn")


def test_blank_self_shot_on_last_shell_still_keeps_turn():
    """Regression: before the fix, if the last shell was blank and the
    shooter fired it at themselves, the recharge branch ran first and
    passed the turn. The blank-self-shot rule must apply regardless of
    whether the chamber ends up empty."""
    e = BuckshotEngine(seed=31)
    e.reset()
    # Give both players enough HP so that reload-on-empty triggers without
    # death. Force the last (and only) remaining shell to be blank.
    e.state.shells = [False]
    for p in e.state.players:
        p.hp = 3
        p.max_hp = 3
    starter = e.state.current_player
    e.step(int(Action.SHOOT_SELF))
    _assert(not e.state.done, "Blank self-shot must not end the game")
    _assert(
        e.state.current_player == starter,
        "Blank self-shot must keep the turn even when it empties the chamber",
    )
    _assert(
        e.state.players[starter].hp == 3,
        "Blank shouldn't damage the shooter",
    )
    print("ok  blank_self_shot_on_last_shell_still_keeps_turn")


def test_live_self_shot_passes_turn_and_damages():
    e = BuckshotEngine(seed=13)
    e.reset()
    e.state.shells = [True, False, True]
    starter = e.state.current_player
    hp_before = e.state.players[starter].hp
    e.step(int(Action.SHOOT_SELF))
    if not e.state.done:
        _assert(e.state.current_player != starter, "Live self-shot must pass turn")
    _assert(e.state.players[starter].hp == hp_before - 1, "Live self-shot must cost 1 HP")
    print("ok  live_self_shot_passes_turn_and_damages")


def test_handcuffs_skip_opponent_turn():
    e = BuckshotEngine(seed=17)
    e.reset()
    e.state.shells = [True, True, False, False]  # ensure no recharge
    p = e.state.players[e.state.current_player]
    p.inventory[int(Item.HANDCUFF)] = 1
    starter = e.state.current_player
    e.step(int(Action.USE_HANDCUFF))
    _assert(e.state.players[1 - starter].skip_next_turn, "Opponent should be cuffed")
    # Now shoot opponent: the would-be opponent's turn is skipped, starter goes again
    e.step(int(Action.SHOOT_OPPONENT))
    if not e.state.done:
        _assert(e.state.current_player == starter, "Cuffed opponent must skip; starter goes again")
        _assert(not e.state.players[1 - starter].skip_next_turn, "Cuff should clear after skip")
    print("ok  handcuffs_skip_opponent_turn")


def test_inverter_flips_next_shell():
    e = BuckshotEngine(seed=19)
    e.reset()
    e.state.shells = [True, False, True]
    p = e.state.players[e.state.current_player]
    p.inventory[int(Item.INVERTER)] = 1
    e.step(int(Action.USE_INVERTER))
    _assert(e.state.shells[0] is False, "Inverter must flip next shell")
    _assert(e.state.shells[1] is False, "Inverter must NOT touch later shells")
    print("ok  inverter_flips_next_shell")


def test_glass_reveals_to_self_only():
    e = BuckshotEngine(seed=23)
    e.reset()
    e.state.shells = [True, False, True]
    starter = e.state.current_player
    p = e.state.players[starter]
    p.inventory[int(Item.GLASS)] = 1
    e.step(int(Action.USE_GLASS))
    _assert(e.state.known_shells[starter].get(0) is True, "Glass must reveal to user")
    _assert(0 not in e.state.known_shells[1 - starter], "Glass must not reveal to opponent")
    print("ok  glass_reveals_to_self_only")


def test_adrenaline_two_step():
    e = BuckshotEngine(seed=29)
    e.reset()
    e.state.shells = [True, False]
    starter = e.state.current_player
    me = e.state.players[starter]
    opp = e.state.players[1 - starter]
    me.inventory[:] = 0
    opp.inventory[:] = 0
    me.inventory[int(Item.ADRENALINE)] = 1
    opp.inventory[int(Item.HANDSAW)] = 1

    mask = e.legal_actions()
    _assert(mask[int(Action.USE_ADRENALINE)], "Adrenaline should be legal when opp has stealable item")
    e.step(int(Action.USE_ADRENALINE))
    _assert(e.state.adrenaline_active, "After USE_ADRENALINE, adrenaline_active must be True")
    _assert(e.state.current_player == starter, "Adrenaline must not pass turn")

    mask2 = e.legal_actions()
    # Only PICK_HANDSAW should be legal (no other items on opp) and shoot actions blocked
    _assert(mask2[int(Action.PICK_HANDSAW)], "PICK_HANDSAW should be legal in adrenaline mode")
    _assert(not mask2[int(Action.SHOOT_OPPONENT)], "Shoots should be blocked while adrenaline_active")

    e.step(int(Action.PICK_HANDSAW))
    _assert(opp.inventory[int(Item.HANDSAW)] == 0, "Stolen item must be removed from opp")
    _assert(e.state.damage_mult == 2, "Stolen handsaw must apply x2 to current player's next shot")
    _assert(not e.state.adrenaline_active, "Adrenaline must clear after pick")
    # Note: the "one non-adrenaline item per turn" flag used to be asserted
    # here, but we now allow chaining items freely within a turn to match
    # the real game, so no such flag exists.
    print("ok  adrenaline_two_step")


def test_pills_can_kill_and_end_game():
    rng = np.random.default_rng(0)
    deaths = 0
    trials = 200
    for trial in range(trials):
        e = BuckshotEngine(seed=int(rng.integers(0, 1_000_000)))
        e.reset()
        starter = e.state.current_player
        e.state.players[starter].hp = 1
        e.state.players[starter].inventory[:] = 0
        e.state.players[starter].inventory[int(Item.PILLS)] = 1
        e.step(int(Action.USE_PILLS))
        if e.state.done:
            deaths += 1
    # Pills are 60% bad → 60% deaths from 1 HP. Allow wide tolerance.
    _assert(80 < deaths < 160, f"Death rate {deaths}/{trials} outside expected ~60%")
    print(f"ok  pills_can_kill_and_end_game (deaths={deaths}/{trials})")


def test_random_vs_random_terminates():
    """Run 200 random-vs-random games; every game must terminate with a winner."""
    rng = np.random.default_rng(123)
    winners = Counter()
    lengths = []
    for trial in range(200):
        e = BuckshotEngine(seed=int(rng.integers(0, 1_000_000)))
        e.reset()
        steps = 0
        while not e.state.done and steps < 1000:
            legal = np.where(e.legal_actions())[0]
            if len(legal) == 0:
                break
            a = int(rng.choice(legal))
            e.step(a)
            steps += 1
        _assert(e.state.done, f"Game {trial} did not terminate in 1000 steps")
        _assert(e.state.winner in (0, 1), f"Bad winner: {e.state.winner}")
        winners[e.state.winner] += 1
        lengths.append(steps)
    # Random play should be roughly balanced (binomial 95% CI is wide for n=200)
    _assert(60 < winners[0] < 140, f"Win imbalance suspicious: {winners}")
    print(
        f"ok  random_vs_random_terminates "
        f"(winners={dict(winners)}, mean_steps={np.mean(lengths):.1f}, "
        f"max_steps={max(lengths)})"
    )


def test_clone_preserves_rng_stream():
    """clone() must yield an engine that produces the EXACT same future
    sequence as the original — verifies we're deep-copying the Generator
    rather than re-seeding from a partial state."""
    e1 = BuckshotEngine(seed=99)
    e1.reset()
    # Burn some randomness so the engine RNG state is mid-stream
    rng_local = np.random.default_rng(0)
    for _ in range(5):
        if e1.state.done:
            break
        legal = np.where(e1.legal_actions())[0]
        e1.step(int(rng_local.choice(legal)))
    e2 = e1.clone()
    # Now play out both with identical action choices and compare every step.
    rng_local = np.random.default_rng(7)
    for _ in range(50):
        if e1.state.done or e2.state.done:
            break
        legal = np.where(e1.legal_actions())[0]
        a = int(rng_local.choice(legal))
        e1.step(a)
        e2.step(a)
        _assert(e1.state.shells == e2.state.shells, "Shells diverged after clone")
        _assert(
            e1.state.players[0].hp == e2.state.players[0].hp
            and e1.state.players[1].hp == e2.state.players[1].hp,
            "HPs diverged after clone",
        )
    print("ok  clone_preserves_rng_stream")


def test_handsaw_survives_beer_reload_mid_turn():
    """Regression: if the player saws, then beers the last shell (forcing a
    mid-turn chamber reload), the saw's damage_mult must persist — the saw
    primed the *next shot*, which still hasn't happened."""
    e = BuckshotEngine(seed=31)
    e.reset()
    # Set up: 1 shell left, player has saw + beer, opp at 2 HP
    e.state.shells = [False]  # blank so next-shot check is clean
    starter = e.state.current_player
    me = e.state.players[starter]
    opp = e.state.players[1 - starter]
    me.inventory[:] = 0
    opp.inventory[:] = 0
    me.inventory[int(Item.HANDSAW)] = 1
    me.inventory[int(Item.BEER)] = 1
    # Saw: damage_mult -> 2
    e.step(int(Action.USE_HANDSAW))
    _assert(e.state.damage_mult == 2, "Saw must set mult=2")
    # Beer ejects last shell → chamber reload
    e.step(int(Action.USE_BEER))
    _assert(e.state.damage_mult == 2,
            f"Saw damage_mult must survive beer-triggered reload (got {e.state.damage_mult})")
    _assert(e.state.current_player == starter, "Beer-after-reload must not pass turn")
    print("ok  handsaw_survives_beer_reload_mid_turn")


def test_handcuff_survives_round_reload_after_last_shot():
    """Regression: cuff applied this turn must still skip the opponent's next
    turn even when the shot that ends the turn also empties the chamber and
    triggers a mid-step reload."""
    e = BuckshotEngine(seed=37)
    e.reset()
    # 1 shell left (live), player has cuff, shoot opp empties chamber
    e.state.shells = [True]
    starter = e.state.current_player
    me = e.state.players[starter]
    opp = e.state.players[1 - starter]
    me.inventory[:] = 0
    opp.inventory[:] = 0
    me.inventory[int(Item.HANDCUFF)] = 1
    opp.hp = 3  # won't die from single live shot
    e.step(int(Action.USE_HANDCUFF))
    _assert(e.state.players[1 - starter].skip_next_turn, "Cuff must flag opponent")
    e.step(int(Action.SHOOT_OPPONENT))
    # After: chamber reloaded (new round), opponent was cuffed, so starter goes again
    _assert(not e.state.done, "Game shouldn't be over")
    _assert(e.state.current_player == starter,
            "Cuffed opponent must skip; starter goes again after reload")
    _assert(not e.state.players[1 - starter].skip_next_turn, "Cuff must clear after consumption")
    print("ok  handcuff_survives_round_reload_after_last_shot")


def test_handcuff_survives_beer_reload_mid_turn():
    """Regression: cuff applied before a beer that empties the chamber must
    persist across the reload — the opponent's next turn hasn't happened yet."""
    e = BuckshotEngine(seed=41)
    e.reset()
    e.state.shells = [False]  # blank last shell
    starter = e.state.current_player
    me = e.state.players[starter]
    opp = e.state.players[1 - starter]
    me.inventory[:] = 0
    opp.inventory[:] = 0
    me.inventory[int(Item.HANDCUFF)] = 1
    me.inventory[int(Item.BEER)] = 1
    e.step(int(Action.USE_HANDCUFF))
    _assert(e.state.players[1 - starter].skip_next_turn, "Cuff must flag opponent")
    e.step(int(Action.USE_BEER))
    _assert(e.state.players[1 - starter].skip_next_turn,
            "Cuff must survive beer-triggered reload")
    _assert(e.state.current_player == starter, "Beer doesn't end turn")
    print("ok  handcuff_survives_beer_reload_mid_turn")


def test_obs_size_matches_documented_layout():
    from rl.engine import _OBS_MAX_SHELLS
    e = BuckshotEngine(seed=0)
    e.reset()
    obs = e.observation(0)
    # 13 scalars + 9 own + 9 opp + _OBS_MAX_SHELLS live + _OBS_MAX_SHELLS blank.
    # Obs width is decoupled from shells_range so checkpoints keep working
    # when chamber sizes are tuned (see the _OBS_MAX_SHELLS definition).
    expected = 13 + 9 + 9 + _OBS_MAX_SHELLS * 2
    _assert(obs.shape == (expected,), f"Obs size {obs.shape} != expected ({expected},)")
    print(f"ok  obs_size_matches_documented_layout (size={obs.shape[0]})")


def main() -> int:
    tests = [
        test_reset_is_deterministic_with_seed,
        test_observation_shape_is_stable,
        test_action_mask_excludes_unowned_items,
        test_smoke_caps_at_max_hp,
        test_item_usable_respects_user_and_target_args,
        test_handsaw_doubles_damage_then_resets,
        test_blank_self_shot_keeps_turn,
        test_blank_self_shot_on_last_shell_still_keeps_turn,
        test_live_self_shot_passes_turn_and_damages,
        test_handcuffs_skip_opponent_turn,
        test_inverter_flips_next_shell,
        test_glass_reveals_to_self_only,
        test_adrenaline_two_step,
        test_pills_can_kill_and_end_game,
        test_random_vs_random_terminates,
        test_clone_preserves_rng_stream,
        test_handsaw_survives_beer_reload_mid_turn,
        test_handcuff_survives_round_reload_after_last_shot,
        test_handcuff_survives_beer_reload_mid_turn,
        test_obs_size_matches_documented_layout,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"ERR  {t.__name__}: {type(e).__name__}: {e}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
