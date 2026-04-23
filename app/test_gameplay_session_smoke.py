"""Smoke tests for the shared rl.engine-backed Telegram session layer.

Run from `app/`:

    python -m test_gameplay_session_smoke
"""

from __future__ import annotations

import os
import sys


APP_DIR = os.path.abspath(os.path.dirname(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from gameplay.actions import SHOOT_OPP_GLYPH, SHOOT_SELF_GLYPH, item_emoji
from gameplay.session import EngineSession
from rl.engine import Action, Item


def _text_for_action(action: Action) -> str:
    if action == Action.SHOOT_OPPONENT:
        return SHOOT_OPP_GLYPH
    if action == Action.SHOOT_SELF:
        return SHOOT_SELF_GLYPH
    return {
        Action.USE_HANDSAW: item_emoji(Item.HANDSAW),
        Action.USE_BEER: item_emoji(Item.BEER),
        Action.USE_SMOKE: item_emoji(Item.SMOKE),
        Action.USE_HANDCUFF: item_emoji(Item.HANDCUFF),
        Action.USE_GLASS: item_emoji(Item.GLASS),
        Action.USE_PHONE: item_emoji(Item.PHONE),
        Action.USE_PILLS: item_emoji(Item.PILLS),
        Action.USE_ADRENALINE: item_emoji(Item.ADRENALINE),
        Action.USE_INVERTER: item_emoji(Item.INVERTER),
    }[action]


def test_pvp_session_starts_and_accepts_current_player_move():
    session = EngineSession.new_pvp(
        first_name="Alice",
        first_chat_id=101,
        second_name="Bob",
        second_chat_id=202,
        seed=7,
    )
    opening = session.start()
    assert opening[101], "first player should receive opening events"
    assert opening[202], "second player should receive opening events"

    current_chat = session.human_chat_ids()[session.state.current_player]
    wrong_chat = session.human_chat_ids()[1 - session.state.current_player]

    wait_notice = session.handle_text(wrong_chat, SHOOT_OPP_GLYPH)
    assert "wait your turn" in wait_notice[wrong_chat][0].text.lower()

    legal = session.engine.legal_actions()
    legal_actions = [Action(i) for i in range(len(legal)) if legal[i]]
    action = next(
        (candidate for candidate in legal_actions if candidate in (Action.SHOOT_OPPONENT, Action.SHOOT_SELF)),
        legal_actions[0],
    )
    dispatch = session.handle_text(current_chat, _text_for_action(action))
    assert dispatch[current_chat], "current player move should emit events"
    print("ok  pvp_session_starts_and_accepts_current_player_move")


class _FakePolicy:
    uses_honest_obs = False

    def initial_hidden(self):
        return None


def test_ai_session_forces_human_first():
    session = EngineSession.new_vs_ai(
        human_name="Karo",
        human_chat_id=303,
        policy=_FakePolicy(),
        seed=11,
    )
    assert session.state.current_player == 0, "AI mode must hand the first move to the human"
    opening = session.start()
    hints = [event.keyboard_hint for event in opening[303]]
    assert "human_turn" in hints, hints
    print("ok  ai_session_forces_human_first")


if __name__ == "__main__":
    test_pvp_session_starts_and_accepts_current_player_move()
    test_ai_session_forces_human_first()
    print("ok  gameplay_session_smoke")
