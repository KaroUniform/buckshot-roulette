"""End-to-end smoke test for the Telegram /ai mode.

No Telegram network traffic — we drive AIRoom directly, check events
for sanity, and assert the game reaches a terminal state under a
legal-action-only player. Run from `app/`:

    python -m test_ai_smoke

Passes when the policy + engine complete 10 games without raising
(the bot won't win all 10 — the random-but-legal human will lose most
of them — but every game should terminate cleanly and the events
should form a coherent stream).
"""

from __future__ import annotations

import os
import random
import sys
import time


APP_DIR = os.path.abspath(os.path.dirname(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# ai/__init__.py handles splicing the repo root in so `from rl.engine ...`
# works — we only need to trigger the import.
import ai  # noqa: F401  pylint: disable=unused-import

from ai.actions import emoji_to_action, item_emoji, pick_emoji, SHOOT_OPP_GLYPH, SHOOT_SELF_GLYPH
from ai.policy import AIPolicy
from ai.room import AIRoom

from rl.engine import Action, Item


def _pick_human_action(room: AIRoom, rng: random.Random) -> Action:
    legal = room.engine.legal_actions()
    legal_actions = [Action(i) for i in range(len(legal)) if legal[i]]
    # Bias slightly toward shooting so games actually end.
    shoots = [a for a in legal_actions if a in (Action.SHOOT_OPPONENT, Action.SHOOT_SELF)]
    if shoots and rng.random() < 0.55:
        return rng.choice(shoots)
    return rng.choice(legal_actions)


def _drive_game(policy: AIPolicy, seed: int) -> dict:
    rng = random.Random(seed)
    room = AIRoom.new(human_name=f"Tester{seed}", policy=policy, seed=seed)
    events = room.start()
    n_events = len(events)
    turns = 0
    while not room.game_over:
        if not room.is_human_turn:
            # AIRoom.start() / step_human should have drained AI events
            # already. A loop here means the engine got stuck.
            raise RuntimeError(
                f"seed={seed}: AI didn't hand control back after "
                f"{n_events} events"
            )
        action = _pick_human_action(room, rng)
        events = room.step_human(action)
        n_events += len(events)
        turns += 1
        if turns > 500:
            raise RuntimeError(f"seed={seed}: game didn't terminate in 500 turns")
    return {
        "seed": seed,
        "n_events": n_events,
        "turns": turns,
        "winner": room.state.winner,
        "human_was": room.human_id,
        "n_reloads": room.state.n_reloads,
    }


def test_actions_roundtrip():
    """USE_* emoji maps back to the correct Action under normal mode."""
    assert emoji_to_action(SHOOT_OPP_GLYPH, adrenaline_active=False) == Action.SHOOT_OPPONENT
    assert emoji_to_action(SHOOT_SELF_GLYPH, adrenaline_active=False) == Action.SHOOT_SELF
    for item, expected in [
        (Item.HANDSAW, Action.USE_HANDSAW),
        (Item.BEER, Action.USE_BEER),
        (Item.ADRENALINE, Action.USE_ADRENALINE),
    ]:
        glyph = item_emoji(item)
        assert emoji_to_action(f"{glyph}x1", adrenaline_active=False) == expected, (
            f"USE map wrong for {item}"
        )
    # PICK mode disambiguates 💉🪚 vs 💉 alone.
    assert emoji_to_action(pick_emoji(Item.HANDSAW), adrenaline_active=True) == Action.PICK_HANDSAW
    # In pick mode, plain 💉 is meaningless — must NOT match USE_ADRENALINE.
    assert emoji_to_action(item_emoji(Item.ADRENALINE), adrenaline_active=True) is None
    print("ok  actions_roundtrip")


def test_policy_loads():
    policy = AIPolicy.get()
    assert policy.obs_dim in (47, 52), f"unexpected obs_dim {policy.obs_dim}"
    print(f"ok  policy_loads (obs_dim={policy.obs_dim}, honest={policy.uses_honest_obs})")


def test_loadout_text_pinned_at_event_time():
    """Regression for bugbot b11b23bd: loadout_text on AIRoomEvent must
    reflect state AT event construction, not a live re-read of
    `room.state` which may have advanced through additional reloads
    before the handler awaits its `send_message`.

    We don't rely on the policy naturally chaining reloads in the
    opening burst (rare, seed-dependent). Instead we construct the
    opening event, snapshot the text, then force the engine through a
    fresh reload and assert the pinned string didn't move.
    """
    policy = AIPolicy.get()
    r = AIRoom.new(human_name="Test", policy=policy, seed=17)
    events = r.start()

    loadout_events = [e for e in events if e.loadout_text is not None]
    assert loadout_events, "start() must emit at least one loadout event"
    pinned_before = loadout_events[0].loadout_text

    # Mutate the live state: pretend a later AI action forced a reload
    # with a different declaration. If the event had been deferred
    # (i.e. rendering at send-time against live state), pinned_before
    # would follow these mutations — it must not.
    r.state.round_initial_live = 99
    r.state.round_initial_blank = 99

    pinned_after = loadout_events[0].loadout_text
    assert pinned_before == pinned_after, (
        f"loadout_text drifted after state mutation: "
        f"{pinned_before!r} -> {pinned_after!r}"
    )
    # And the pinned string must NOT contain the injected 99×99 values.
    assert "99" not in pinned_before, (
        f"pinned loadout leaked live state: {pinned_before!r}"
    )
    print(f"ok  loadout_text_pinned_at_event_time  (pinned={pinned_before!r})")


def test_glass_phone_reveal_to_human():
    """Regression for bugbot 2e3d1a25: USE_GLASS / USE_PHONE captions
    must surface the `info` dict contents to the human — otherwise the
    items are visually useless. AI captions stay generic so we don't
    leak the AI's private knowledge across the table.
    """
    from ai.render import action_caption

    # Human side — both kinds of reveals should be visible.
    human_live = action_caption(int(Action.USE_GLASS), {"glass": "live"}, actor="human")
    assert "live" in human_live and "💥" in human_live, human_live
    human_blank = action_caption(int(Action.USE_GLASS), {"glass": "blank"}, actor="human")
    assert "blank" in human_blank and "🫧" in human_blank, human_blank

    human_phone = action_caption(int(Action.USE_PHONE), {"phone": (3, "live")}, actor="human")
    # Position is 1-indexed for display: pos=3 → "#4".
    assert "#4" in human_phone and "live" in human_phone, human_phone

    # AI side — no private info should leak.
    ai_glass = action_caption(int(Action.USE_GLASS), {"glass": "live"}, actor="ai")
    assert "live" not in ai_glass and "blank" not in ai_glass, ai_glass
    ai_phone = action_caption(int(Action.USE_PHONE), {"phone": (3, "live")}, actor="ai")
    assert "#4" not in ai_phone and "live" not in ai_phone, ai_phone

    # Missing info key shouldn't crash (defensive fallback to generic text).
    human_glass_missing = action_caption(int(Action.USE_GLASS), {}, actor="human")
    assert "inspected" in human_glass_missing
    print("ok  glass_phone_reveal_to_human")


def test_terminal_events_use_game_over_keyboard():
    """Regression for bugbot 0c83ef58: when the human's action ends the
    game, every event in the resulting list must carry
    keyboard_hint='game_over'. Otherwise the handler asks
    `human_turn_keyboard` for a done-state keyboard which is empty →
    Telegram rejects the reply_markup.
    """
    policy = AIPolicy.get()
    # Hunt through seeds until we find one where the human is the first
    # mover and a single SHOOT_OPPONENT on seed-0 shells lands as a lethal
    # live shot — saves us from mutating engine internals.
    r = AIRoom.new(human_name="Test", policy=policy, seed=7)
    r.start()
    # Force the scenario: set opponent HP to 1 so any live shot is lethal,
    # and make sure it's the human's turn with a live shell up next.
    while not r.is_human_turn and not r.game_over:
        # Drain AI turns until control returns; AIRoom.start already did
        # an opening drain so this is usually a no-op.
        break
    assert r.is_human_turn, "seed=7 should hand the opening to the human"
    r.state.players[r.ai_id].hp = 1
    # Ensure chamber has at least one live next; if the top is blank,
    # swap positions so the next shot is lethal. Engine represents shells
    # as a list of bools where True=live.
    if not r.state.shells or not r.state.shells[0]:
        for i, s in enumerate(r.state.shells):
            if s:
                r.state.shells[0], r.state.shells[i] = r.state.shells[i], r.state.shells[0]
                break
    assert r.state.shells and r.state.shells[0], "need a live shell on top"

    events = r.step_human(Action.SHOOT_OPPONENT)
    assert r.game_over, "shot should have ended the game"
    bad = [e for e in events if e.keyboard_hint != "game_over"]
    assert not bad, (
        f"every event after a kill-shot must hint game_over, got: "
        f"{[(e.text[:40], e.keyboard_hint) for e in events]}"
    )
    print(f"ok  terminal_events_use_game_over_keyboard  ({len(events)} events)")


def test_adrenaline_fallback_shoot_parses():
    """Regression for bugbot 3286fcd8: when adrenaline is active but no
    picks are usable, the engine falls back to making SHOOT_OPPONENT /
    SHOOT_SELF legal and the render layer emits 🔼/🔽 buttons. The parser
    must accept those glyphs in pick-mode or the user is stuck pressing
    visible buttons that produce 'Make a valid move' forever.
    """
    assert emoji_to_action(SHOOT_OPP_GLYPH, adrenaline_active=True) == Action.SHOOT_OPPONENT
    assert emoji_to_action(SHOOT_SELF_GLYPH, adrenaline_active=True) == Action.SHOOT_SELF
    # And pick tokens still win when both are present (longest-match
    # isn't needed here because prefixes don't collide, but verify).
    assert (
        emoji_to_action(pick_emoji(Item.HANDSAW), adrenaline_active=True)
        == Action.PICK_HANDSAW
    )
    print("ok  adrenaline_fallback_shoot_parses")


def test_ai_terminal_events_use_game_over_keyboard():
    """Regression for bugbot 92f6089d: when the AI's action ends the
    game, the kill-shot caption must NOT carry keyboard_hint='wait'.
    That hint makes `_send_events` paint the '🕓AI is thinking🕓'
    keyboard and sleep 1.1s before the game-over message — a misleading
    thinking indicator after the game is already decided.
    """
    policy = AIPolicy.get()
    # Force a scenario: human goes first, immediately USE_HANDCUFF to
    # skip to AI (if handcuff is available) — but the simplest way is
    # to directly construct a situation where the AI shoots last.
    # Cheap trick: hunt for a seed where AI goes first and the human HP
    # is already 1 after a mutation, then let the AI drain.
    for seed in range(100):
        r = AIRoom.new(human_name="Test", policy=policy, seed=seed)
        if r.is_human_turn:
            continue  # need AI-first seeds
        # Drop human to 1 HP and ensure the next shell is live so the AI
        # (which often shoots opponent on a known-live) ends the game.
        r.state.players[r.human_id].hp = 1
        if r.state.shells:
            r.state.shells[0] = True
        events = r.start()
        if r.game_over:
            # Earlier AI sub-actions (cuff, phone, etc.) legitimately
            # carry 'wait' because they were intermediate. The bug is
            # specifically that the step which SETS done=True has
            # keyboard_hint='wait' — triggering the 1.1s "thinking"
            # pause between the kill-shot and the game-over message.
            # The last two events must be the terminal action caption
            # and the game-over event, both hinting 'game_over'.
            assert events[-1].keyboard_hint == "game_over", (
                f"final event should be game_over, got {events[-1]!r}"
            )
            assert events[-2].keyboard_hint == "game_over", (
                f"penultimate event (the terminal AI action) should be "
                f"game_over, got: {events[-2]!r}"
            )
            print(f"ok  ai_terminal_events_use_game_over_keyboard  "
                  f"(seed={seed}, {len(events)} events)")
            return
    # If no seed produced an AI kill-shot after 100 tries the policy is
    # behaving oddly — surface that rather than silently passing.
    raise AssertionError(
        "couldn't find a seed where a mutated HP=1 human dies to an AI "
        "opening burst; investigate if this fires in CI"
    )


def test_games_terminate():
    policy = AIPolicy.get()
    ai_wins = 0
    total = 10
    t0 = time.perf_counter()
    for seed in range(total):
        r = _drive_game(policy, seed)
        if r["winner"] != r["human_was"]:
            ai_wins += 1
    dt = time.perf_counter() - t0
    # Against a random-legal human the AI should be heavily favored.
    # We don't assert a specific win rate (it's stochastic), just that
    # games terminate.
    print(f"ok  games_terminate  ({total} games in {dt:.1f}s, ai_wins={ai_wins}/{total})")


def main() -> int:
    tests = [
        test_actions_roundtrip,
        test_policy_loads,
        test_loadout_text_pinned_at_event_time,
        test_glass_phone_reveal_to_human,
        test_terminal_events_use_game_over_keyboard,
        test_adrenaline_fallback_shoot_parses,
        test_ai_terminal_events_use_game_over_keyboard,
        test_games_terminate,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # pragma: no cover
            print(f"FAIL {t.__name__}: {exc}")
            failed += 1
            import traceback; traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
