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


def test_reload_banner_renders_inventories_and_pause():
    """Regression for the UX ask: at round boundaries the reload event
    must consolidate into a single message (banner + loadout + both
    inventories) AND request a longer pause so the player can read it
    before the AI's next burst arrives.
    """
    from ai.render import reload_banner
    from ai.room import RELOAD_PAUSE_MS

    # Legacy zero-arg form still works (used as a fallback).
    assert "new round" in reload_banner().lower()

    policy = AIPolicy.get()
    r = AIRoom.new(human_name="Tester", policy=policy, seed=3)
    r.start()
    # Force specific declaration values so the assertions are stable
    # regardless of the seeded RNG's choice.
    s = r.state
    s.round_initial_live = 2
    s.round_initial_blank = 3
    banner = reload_banner(s, r.human_name, r.human_id)

    assert "Tester" in banner, banner
    assert "🤖 AI" in banner, banner
    assert "💥×2" in banner and "🫧×3" in banner, banner
    # Multi-line with a visible separator.
    assert "━" in banner, banner
    assert banner.count("\n") >= 4, banner

    # Pause constant must be meaningfully longer than the default
    # inter-action beat (1100 ms) — otherwise round boundaries feel
    # indistinguishable from a blank self-shot.
    assert RELOAD_PAUSE_MS >= 2000, RELOAD_PAUSE_MS
    print(f"ok  reload_banner_renders_inventories_and_pause  "
          f"(banner={banner.splitlines()[1]!r}, pause={RELOAD_PAUSE_MS}ms)")


def test_pick_action_captions_include_outcome():
    """Regression for the "куда пропал один выстрел?" report: PICK_<item>
    captions used to say only "💉 stole your X and is using it" — the
    item's actual effect (ejected shell from beer, pills gain/loss, etc.)
    was silently dropped, so a player watching the AI stole-and-use a
    beer saw no shell-ejection message and thought a round had gone
    missing.
    """
    from ai.render import action_caption

    # PICK_BEER must reveal the ejected shell (public info — both
    # players see the shell fly out of the shotgun).
    cap = action_caption(int(Action.PICK_BEER), {"beer_ejected": "live"}, actor="ai")
    assert "💥" in cap and "flew out" in cap, cap
    cap = action_caption(int(Action.PICK_BEER), {"beer_ejected": "blank"}, actor="human")
    assert "🫧" in cap and "flew out" in cap, cap

    # PICK_PILLS must reveal good/bad (public — visible via HP bar).
    cap = action_caption(int(Action.PICK_PILLS), {"pills": "good"}, actor="ai")
    assert "healed 2" in cap, cap
    cap = action_caption(int(Action.PICK_PILLS), {"pills": "bad"}, actor="ai")
    assert "lost 1" in cap, cap

    # PICK_HANDSAW / PICK_HANDCUFF / PICK_SMOKE / PICK_INVERTER
    # always have the same observable outcome.
    cap = action_caption(int(Action.PICK_HANDSAW), {}, actor="ai")
    assert "2×" in cap, cap
    cap = action_caption(int(Action.PICK_HANDCUFF), {}, actor="ai")
    assert "skip" in cap, cap
    cap = action_caption(int(Action.PICK_SMOKE), {}, actor="ai")
    assert "healed 1" in cap, cap
    cap = action_caption(int(Action.PICK_INVERTER), {}, actor="ai")
    assert "polarity" in cap, cap

    # PICK_GLASS / PICK_PHONE reveal private info — human picker sees
    # the result, AI picker stays opaque.
    cap = action_caption(int(Action.PICK_GLASS), {"glass": "live"}, actor="human")
    assert "💥" in cap, cap
    cap = action_caption(int(Action.PICK_GLASS), {"glass": "live"}, actor="ai")
    assert "💥" not in cap and "live" not in cap, cap
    cap = action_caption(int(Action.PICK_PHONE), {"phone": (2, "blank")}, actor="human")
    assert "#3" in cap and "blank" in cap, cap
    cap = action_caption(int(Action.PICK_PHONE), {"phone": (2, "blank")}, actor="ai")
    assert "#3" not in cap and "blank" not in cap, cap

    print("ok  pick_action_captions_include_outcome")


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


def test_human_action_passing_turn_uses_wait_hint():
    """Regression for bugbot 50bfb404: when the human's (non-terminal)
    action passes control to the AI — e.g. shooting the opponent with
    a live shell that doesn't kill — the human-event hints must be
    'wait', not 'human_turn'. Otherwise the handler calls
    `human_turn_keyboard` with `state.current_player=ai_id` and paints
    a legality-mangled keyboard until the next event replaces it.

    We test `_compose_human_events` directly so we observe the hints
    BEFORE `_drain_ai` runs and flips the turn back. In production this
    matters because the handler sends each event with its own keyboard
    before the next event overwrites it.
    """
    policy = AIPolicy.get()
    # Find a seed where the human goes first, then step the engine
    # manually so we can inspect `_compose_human_events` output with
    # state.current_player = ai_id and done = False.
    for seed in range(100):
        r = AIRoom.new(human_name="Test", policy=policy, seed=seed)
        if not r.is_human_turn:
            continue
        r.state.players[r.ai_id].hp = 2  # survive the shot
        if not r.state.shells:
            continue
        r.state.shells[0] = True  # live on top
        legal = r.engine.legal_actions()
        if not legal[int(Action.SHOOT_OPPONENT)]:
            continue
        prev_reloads = r.state.n_reloads
        _, _, done, info = r.engine.step(int(Action.SHOOT_OPPONENT))
        if done:
            continue
        if r.is_human_turn:
            # Still the human (e.g. engine granted keep-turn via some
            # item state we didn't foresee). We need the turn to flip.
            continue
        events = r._compose_human_events(
            Action.SHOOT_OPPONENT, info, prev_reloads, done,
        )
        bad = [e for e in events if e.keyboard_hint != "wait"]
        assert not bad, (
            f"all human-event hints should be 'wait' after turn passes, "
            f"got: {[(e.text[:50], e.keyboard_hint) for e in events]}"
        )
        print(f"ok  human_action_passing_turn_uses_wait_hint  "
              f"(seed={seed}, {len(events)} events)")
        return
    raise AssertionError(
        "couldn't find a seed that reproduces a non-terminal human-to-AI "
        "turn transition — investigate if this fires in CI"
    )


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


def test_ai_action_counter_increments_during_drain():
    """Regression: `n_ai_actions` must count every engine step the AI
    takes inside `_drain_ai`, not stay at zero. Bugbot flagged that the
    DB column previously named `n_turns` stored only human turns — the
    split-counter fix relies on this counter actually moving.
    """
    policy = AIPolicy.get()
    # Drive a full game with random-legal human. The AI will take at
    # least one action (opening or response); count must end > 0.
    seed = 0
    rng = random.Random(seed)
    room = AIRoom.new(human_name="Tester", policy=policy, seed=seed)
    room.start()
    while not room.game_over:
        if not room.is_human_turn:
            raise RuntimeError("AI didn't return control")
        # Any legal action, bias to shoots so the game ends.
        legal = room.engine.legal_actions()
        legal_actions = [Action(i) for i in range(len(legal)) if legal[i]]
        shoots = [a for a in legal_actions if a in (Action.SHOOT_OPPONENT, Action.SHOOT_SELF)]
        action = rng.choice(shoots) if shoots and rng.random() < 0.55 else rng.choice(legal_actions)
        room.step_human(action)
    assert room.n_ai_actions > 0, (
        f"AI never acted during the game? n_ai_actions={room.n_ai_actions}"
    )
    assert room.n_human_turns > 0, room.n_human_turns
    print(f"ok  ai_action_counter_increments_during_drain  "
          f"(human={room.n_human_turns}, ai={room.n_ai_actions})")


def test_ai_room_captures_first_mover():
    """Regression for bugbot: `human_went_first` must reflect the actual
    engine-randomised starting player, not `human_id == 0`. The engine
    uses numpy PCG64 for `current_player`, AIRoom uses Mersenne Twister
    for `human_id`; seeding both with the same int gives uncorrelated
    picks, so the old slot-based heuristic was wrong ~half the time.
    """
    policy = AIPolicy.get()
    # Scan enough seeds that both "slot heuristic agrees with truth"
    # and "slot heuristic disagrees with truth" cases actually appear —
    # otherwise the test would silently pass on a fortuitous subset.
    saw_agree = False
    saw_disagree = False
    for seed in range(50):
        r = AIRoom.new(human_name="Test", policy=policy, seed=seed)
        truth = r.engine.state.current_player == r.human_id
        assert r.human_went_first == truth, (
            f"seed={seed}: captured={r.human_went_first}, truth={truth} "
            f"(current_player={r.engine.state.current_player}, "
            f"human_id={r.human_id})"
        )
        slot_heuristic = r.human_id == 0
        if slot_heuristic == truth:
            saw_agree = True
        else:
            saw_disagree = True
    assert saw_agree and saw_disagree, (
        "50 seeds didn't produce both branches — widen the range "
        "before declaring the regression covered"
    )
    print("ok  ai_room_captures_first_mover")


def test_stats_store_roundtrip():
    """Persist a few synthetic matches and assert summary + leaderboard
    line up with the raw inserts. Uses a temp-file DB so WAL / schema
    behavior matches production; in-memory SQLite has per-connection
    DBs which would defeat the point.
    """
    import asyncio
    import tempfile
    from datetime import datetime, timezone
    from ai import stats as _stats

    path = tempfile.mktemp(suffix=".sqlite3")
    try:
        async def scenario():
            store = _stats.StatsStore(path)
            await store.init_schema()
            now = datetime.now(timezone.utc)
            # 3 matches: alice beats the AI once in 2 tries, bob loses.
            records = [
                _stats.MatchRecord(
                    ended_at=now, chat_id=100, user_id=1, human_name="Alice",
                    ai_won=True, human_went_first=True,
                    n_human_turns=14, n_ai_actions=12,
                    n_reloads=3, seed=42, duration_ms=45000,
                ),
                _stats.MatchRecord(
                    ended_at=now, chat_id=100, user_id=1, human_name="Alice",
                    ai_won=False, human_went_first=False,
                    n_human_turns=9, n_ai_actions=11,
                    n_reloads=2, seed=43, duration_ms=30000,
                ),
                _stats.MatchRecord(
                    ended_at=now, chat_id=200, user_id=2, human_name="Bob",
                    ai_won=True, human_went_first=True,
                    n_human_turns=11, n_ai_actions=9,
                    n_reloads=2, seed=44, duration_ms=40000,
                ),
            ]
            for r in records:
                await store.record(r)
            summary = await store.summary()
            assert summary.total == 3, summary
            assert summary.ai_wins == 2, summary
            assert summary.human_wins == 1, summary
            # `last_total_moves` must be SUM of human+ai counters,
            # not just human turns — the last insert was Bob's
            # 11 + 9 = 20.
            assert summary.last_total_moves == 20, summary
            # Leaderboard: only Alice (Bob has 0 wins vs AI).
            top = await store.top_humans(limit=5)
            assert len(top) == 1, top
            assert top[0].name == "Alice"
            assert top[0].wins == 1 and top[0].games == 2
        asyncio.run(scenario())
    finally:
        for suffix in ("", "-wal", "-shm"):
            p = path + suffix
            if os.path.exists(p):
                os.remove(p)
    print("ok  stats_store_roundtrip")


def test_wilson_ci_edges():
    from ai.stats import wilson_ci
    # Zero trials — degenerate but must not crash.
    assert wilson_ci(0, 0) == (0.0, 0.0)
    # All successes on N=10 — upper bound is 1.0, lower is < 1.0
    lo, hi = wilson_ci(10, 10)
    assert 0 < lo < 1 and hi == 1.0, (lo, hi)
    # Balanced coin, N=100 — interval straddles 0.5
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi, (lo, hi)
    # Monotonicity: more data → tighter interval
    lo10, hi10 = wilson_ci(9, 10)
    lo100, hi100 = wilson_ci(90, 100)
    assert (hi100 - lo100) < (hi10 - lo10)
    print("ok  wilson_ci_edges")


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
        test_reload_banner_renders_inventories_and_pause,
        test_pick_action_captions_include_outcome,
        test_terminal_events_use_game_over_keyboard,
        test_human_action_passing_turn_uses_wait_hint,
        test_adrenaline_fallback_shoot_parses,
        test_ai_terminal_events_use_game_over_keyboard,
        test_wilson_ci_edges,
        test_ai_action_counter_increments_during_drain,
        test_ai_room_captures_first_mover,
        test_stats_store_roundtrip,
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
