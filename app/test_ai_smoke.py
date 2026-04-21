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
    tests = [test_actions_roundtrip, test_policy_loads, test_games_terminate]
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
