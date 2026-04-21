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


def test_handsaw_does_not_stack():
    """Real Buckshot Roulette caps next-shot damage at 2x. A second
    handsaw before firing must be a no-op — neither the damage_mult
    nor the USE_HANDSAW legality should escalate past 2. Absence of
    this cap would let the RL agent discover and exploit nonexistent
    mechanics during self-play.
    """
    e = BuckshotEngine(seed=7)
    e.reset()
    e.state.shells = [True, False]
    p = e.state.players[e.state.current_player]
    p.inventory[int(Item.HANDSAW)] = 2
    e.step(int(Action.USE_HANDSAW))
    _assert(e.state.damage_mult == 2, "first saw must set damage_mult=2")
    mask = e.legal_actions()
    _assert(not mask[int(Action.USE_HANDSAW)],
            "second USE_HANDSAW must be illegal while damage_mult already 2")
    # _item_usable should also report False directly
    _assert(not e._item_usable(Item.HANDSAW, p, e.state.players[1 - e.state.current_player]),
            "_item_usable(HANDSAW) must reject second saw")
    # And the mult stays at 2 (no stacking even if a buggy caller forces it)
    _assert(e.state.damage_mult == 2, "damage_mult must not escalate")
    print("ok  handsaw_does_not_stack")


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


def test_inverter_does_not_reveal_to_anyone():
    # Game rule: the inverter flips the chambered shell's polarity but does
    # NOT show the new state to the user — the shell stays in the barrel.
    # So if slot 0 was unknown, it remains unknown to BOTH players.
    e = BuckshotEngine(seed=101)
    e.reset()
    e.state.shells = [True, False, True]
    user = e.state.current_player
    e.state.known_shells[user].pop(0, None)
    e.state.known_shells[1 - user].pop(0, None)
    p = e.state.players[user]
    p.inventory[int(Item.INVERTER)] = 1
    e.step(int(Action.USE_INVERTER))
    _assert(
        0 not in e.state.known_shells[user],
        "Inverter must NOT reveal slot 0 to the user",
    )
    _assert(
        0 not in e.state.known_shells[1 - user],
        "Inverter must NOT reveal slot 0 to the opponent",
    )

    # But existing knowledge about slot 0 on EITHER side must flip to stay valid.
    e2 = BuckshotEngine(seed=102)
    e2.reset()
    e2.state.shells = [False, True, False]
    user2 = e2.state.current_player
    e2.state.known_shells[user2][0] = False
    e2.state.known_shells[1 - user2][0] = False
    p2 = e2.state.players[user2]
    p2.inventory[int(Item.INVERTER)] = 1
    e2.step(int(Action.USE_INVERTER))
    _assert(
        e2.state.known_shells[user2].get(0) is True,
        "User's prior knowledge of slot 0 must flip after inversion",
    )
    _assert(
        e2.state.known_shells[1 - user2].get(0) is True,
        "Opponent's prior knowledge of slot 0 must flip after inversion",
    )
    print("ok  inverter_does_not_reveal_to_anyone")


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


def test_clone_preserves_honest_obs_flag():
    """Regression: clone() uses __new__ and copies fields explicitly, so it
    must remember to carry honest_obs. Otherwise observation() raises
    AttributeError on the cloned engine."""
    for flag in (False, True):
        e1 = BuckshotEngine(seed=42, honest_obs=flag)
        e1.reset()
        e2 = e1.clone()
        _assert(e2.honest_obs == flag, f"honest_obs={flag} must survive clone")
        obs1 = e1.observation(e1.state.current_player)
        obs2 = e2.observation(e2.state.current_player)
        _assert(obs1.shape == obs2.shape,
                f"clone obs shape diverges under honest_obs={flag}")
        _assert(np.array_equal(obs1, obs2),
                f"clone obs content diverges under honest_obs={flag}")
    print("ok  clone_preserves_honest_obs_flag")


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


def test_honest_obs_size_and_layout():
    """honest_obs=True yields 52-dim obs (47 - 2 removed + 7 added).

    Specifically: no n_live/n_blank slots; instead the seven per-round public
    event counters sit in the core section. Invariant: counters start at zero
    immediately after reset (no shots have been fired yet).
    """
    from rl.engine import _OBS_MAX_SHELLS
    e = BuckshotEngine(seed=0, honest_obs=True)
    e.reset()
    obs = e.observation(0)
    # 18 scalars + 9 own + 9 opp + 2*_OBS_MAX_SHELLS
    expected = 18 + 9 + 9 + _OBS_MAX_SHELLS * 2
    _assert(obs.shape == (expected,), f"Honest obs shape {obs.shape} != ({expected},)")
    # Default (hack) obs should still be 47-dim.
    e2 = BuckshotEngine(seed=0)
    e2.reset()
    _assert(e2.observation(0).shape == (47,), "Default obs must stay 47-dim")
    print(f"ok  honest_obs_size_and_layout (honest={obs.shape[0]}, hack={e2.observation(0).shape[0]})")


def test_honest_obs_initial_declaration_set_after_reset():
    """After reset, round_initial_live/blank must match the loaded chamber,
    and all per-round event counters must be zero."""
    e = BuckshotEngine(seed=3, honest_obs=True)
    s = e.reset()
    n_live_actual = sum(1 for x in s.shells if x)
    n_blank_actual = len(s.shells) - n_live_actual
    _assert(s.round_initial_live == n_live_actual,
            f"round_initial_live={s.round_initial_live} != actual {n_live_actual}")
    _assert(s.round_initial_blank == n_blank_actual,
            f"round_initial_blank={s.round_initial_blank} != actual {n_blank_actual}")
    for f in ("shots_fired_live_this_round", "shots_fired_blank_this_round",
              "beer_ejected_live_this_round", "beer_ejected_blank_this_round",
              "inverter_uses_this_round"):
        _assert(getattr(s, f) == 0, f"{f} must start at 0")
    print("ok  honest_obs_initial_declaration_set_after_reset")


def test_honest_obs_shot_counters_increment():
    """Every shot must bump exactly one of shots_fired_{live,blank} by 1."""
    e = BuckshotEngine(seed=0, honest_obs=True)
    s = e.reset()
    # Rig chamber: live-blank-live-blank-live (odd positions)
    s.shells = [True, False, True, False, True]
    s.round_initial_live = 3
    s.round_initial_blank = 2
    s.shots_fired_live_this_round = 0
    s.shots_fired_blank_this_round = 0
    starter = s.current_player
    # Shoot opponent (live)
    e.step(int(Action.SHOOT_OPPONENT))
    _assert(s.shots_fired_live_this_round == 1,
            f"after live shot: live_count={s.shots_fired_live_this_round}")
    _assert(s.shots_fired_blank_this_round == 0, "blank_count must stay 0")
    # Next shell is blank; whoever's turn it is, shoot self (keeps turn)
    actor = s.current_player
    e.step(int(Action.SHOOT_SELF))
    _assert(s.shots_fired_blank_this_round == 1,
            f"after blank self-shot: blank_count={s.shots_fired_blank_this_round}")
    _assert(s.shots_fired_live_this_round == 1, "live_count must stay at 1")
    print(f"ok  honest_obs_shot_counters_increment (starter={starter}, after_blank={actor})")


def test_honest_obs_beer_counter_increments():
    e = BuckshotEngine(seed=0, honest_obs=True)
    s = e.reset()
    s.shells = [False, True, True, False]  # blank first so beer ejects a blank
    s.round_initial_live = 2
    s.round_initial_blank = 2
    s.shots_fired_live_this_round = 0
    s.shots_fired_blank_this_round = 0
    s.beer_ejected_live_this_round = 0
    s.beer_ejected_blank_this_round = 0
    s.players[s.current_player].inventory[int(Item.BEER)] = 1
    e.step(int(Action.USE_BEER))
    _assert(s.beer_ejected_blank_this_round == 1,
            f"beer ejected blank, counter={s.beer_ejected_blank_this_round}")
    _assert(s.beer_ejected_live_this_round == 0, "live beer counter must stay 0")
    # Second beer ejects a live shell
    s.players[s.current_player].inventory[int(Item.BEER)] = 1
    e.step(int(Action.USE_BEER))
    _assert(s.beer_ejected_live_this_round == 1,
            f"beer ejected live, counter={s.beer_ejected_live_this_round}")
    _assert(s.beer_ejected_blank_this_round == 1, "blank beer counter must hold at 1")
    print("ok  honest_obs_beer_counter_increments")


def test_honest_obs_inverter_counter_increments():
    e = BuckshotEngine(seed=0, honest_obs=True)
    s = e.reset()
    s.shells = [True, False, True]
    s.inverter_uses_this_round = 0
    s.players[s.current_player].inventory[int(Item.INVERTER)] = 1
    e.step(int(Action.USE_INVERTER))
    _assert(s.inverter_uses_this_round == 1, f"inverter count={s.inverter_uses_this_round}")
    # Second inverter on same shell flips it back; counter still climbs.
    s.players[s.current_player].inventory[int(Item.INVERTER)] = 1
    e.step(int(Action.USE_INVERTER))
    _assert(s.inverter_uses_this_round == 2, f"inverter count={s.inverter_uses_this_round}")
    print("ok  honest_obs_inverter_counter_increments")


def test_honest_obs_counters_reset_on_new_round():
    """When the chamber empties and a fresh round loads, all per-round counters
    must reset to zero and the initial declaration must reflect the NEW round."""
    e = BuckshotEngine(seed=0, honest_obs=True)
    s = e.reset()
    s.shells = [False]  # single blank; self-shot keeps turn AND forces reload
    s.round_initial_live = 0
    s.round_initial_blank = 1
    s.shots_fired_live_this_round = 0
    s.shots_fired_blank_this_round = 0
    e.step(int(Action.SHOOT_SELF))
    # Chamber emptied → _load_round ran → counters reset; initial L/B reflect new round.
    _assert(s.shots_fired_live_this_round == 0, "shots_fired_live must reset")
    _assert(s.shots_fired_blank_this_round == 0, "shots_fired_blank must reset")
    _assert(s.round_initial_live + s.round_initial_blank == len(s.shells),
            f"initial L+B ({s.round_initial_live}+{s.round_initial_blank}) != new chamber len {len(s.shells)}")
    actual_live = sum(1 for x in s.shells if x)
    _assert(s.round_initial_live == actual_live,
            f"round_initial_live={s.round_initial_live} vs actual_live={actual_live}")
    print("ok  honest_obs_counters_reset_on_new_round")


def test_honest_obs_is_symmetric_across_players():
    """All 7 per-round public counters sit at the same slot indices in both
    players' observations (they're public info, so obs(0) and obs(1) agree
    on those slots). Private fields (known_shells) may differ."""
    e = BuckshotEngine(seed=4, honest_obs=True)
    s = e.reset()
    # Fire a couple of shots to exercise the counters.
    for _ in range(3):
        if s.done:
            break
        legal = np.where(e.legal_actions())[0]
        if len(legal) == 0:
            break
        # Prefer shoot-opponent/self over items to make the test deterministic.
        if int(Action.SHOOT_OPPONENT) in legal:
            e.step(int(Action.SHOOT_OPPONENT))
        else:
            e.step(int(legal[0]))
    o0 = e.observation(0)
    o1 = e.observation(1)
    # Honest-obs layout: core[11..17] are the 7 public counters.
    _assert(np.array_equal(o0[11:18], o1[11:18]),
            f"public counters must match across players: {o0[11:18]} vs {o1[11:18]}")
    print(f"ok  honest_obs_is_symmetric_across_players (counters={o0[11:18].tolist()})")


def test_honest_obs_respects_shell_invariant_under_no_inverter():
    """Without inverter uses, the true chamber L/B equals:
       initial_L - shots_fired_live - beer_ejected_live (same for blanks).
    This is the invariant a recurrent policy's internal belief would maintain.
    """
    e = BuckshotEngine(seed=7, honest_obs=True)
    s = e.reset()
    # Drive random legal actions, avoiding INVERTER to preserve the invariant.
    rng = np.random.default_rng(123)
    for _ in range(80):
        if s.done:
            break
        mask = e.legal_actions()
        mask[int(Action.USE_INVERTER)] = False
        mask[int(Action.PICK_INVERTER)] = False
        legal = np.where(mask)[0]
        if len(legal) == 0:
            break
        e.step(int(rng.choice(legal)))
        # Invariant must hold every step (within the current round).
        true_live = sum(1 for x in s.shells if x)
        true_blank = len(s.shells) - true_live
        derived_live = (s.round_initial_live
                        - s.shots_fired_live_this_round
                        - s.beer_ejected_live_this_round)
        derived_blank = (s.round_initial_blank
                         - s.shots_fired_blank_this_round
                         - s.beer_ejected_blank_this_round)
        _assert(derived_live == true_live and derived_blank == true_blank,
                f"invariant broken: derived=({derived_live},{derived_blank})"
                f" actual=({true_live},{true_blank})"
                f" initial=({s.round_initial_live},{s.round_initial_blank})"
                f" shots=({s.shots_fired_live_this_round},{s.shots_fired_blank_this_round})"
                f" beer=({s.beer_ejected_live_this_round},{s.beer_ejected_blank_this_round})")
    print("ok  honest_obs_respects_shell_invariant_under_no_inverter")


def main() -> int:
    tests = [
        test_reset_is_deterministic_with_seed,
        test_observation_shape_is_stable,
        test_action_mask_excludes_unowned_items,
        test_smoke_caps_at_max_hp,
        test_item_usable_respects_user_and_target_args,
        test_handsaw_doubles_damage_then_resets,
        test_handsaw_does_not_stack,
        test_blank_self_shot_keeps_turn,
        test_blank_self_shot_on_last_shell_still_keeps_turn,
        test_live_self_shot_passes_turn_and_damages,
        test_handcuffs_skip_opponent_turn,
        test_inverter_flips_next_shell,
        test_inverter_does_not_reveal_to_anyone,
        test_glass_reveals_to_self_only,
        test_adrenaline_two_step,
        test_pills_can_kill_and_end_game,
        test_random_vs_random_terminates,
        test_clone_preserves_rng_stream,
        test_handsaw_survives_beer_reload_mid_turn,
        test_handcuff_survives_round_reload_after_last_shot,
        test_handcuff_survives_beer_reload_mid_turn,
        test_obs_size_matches_documented_layout,
        test_honest_obs_size_and_layout,
        test_honest_obs_initial_declaration_set_after_reset,
        test_honest_obs_shot_counters_increment,
        test_honest_obs_beer_counter_increments,
        test_honest_obs_inverter_counter_increments,
        test_honest_obs_counters_reset_on_new_round,
        test_honest_obs_is_symmetric_across_players,
        test_honest_obs_respects_shell_invariant_under_no_inverter,
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
