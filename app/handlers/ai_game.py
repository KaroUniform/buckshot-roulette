"""Telegram handler for the single-player AI mode.

Routing flow:
    /ai                       → start a new AIRoom (lazy-loads E19)
    Text during `in_ai_game`  → forwarded to AIRoom.step_human
    /leave or 🚪Leave         → shared with the multiplayer handler;
                                this module registers no leave logic

The AI policy is imported lazily inside handlers so the bot still
boots in environments without torch / without the checkpoint. The
error is caught and surfaced to the user as a one-line message.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict

from aiogram import Bot, F, Router, types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from core.game_states import GameStates

logger = logging.getLogger(__name__)

router = Router()


# chat_id → AIRoom. Kept in-process; rooms evaporate on restart, which
# is fine for single-player (no one else loses if a game is lost).
_AI_ROOMS: Dict[int, "AIRoom"] = {}


def _send_keyboard(hint: str, room: "AIRoom | None"):
    """Build the ReplyKeyboardMarkup for an event."""
    from ai import render
    if hint == "wait":
        return render.wait_keyboard()
    if hint == "game_over":
        rematch_kb = [
            [types.KeyboardButton(text="🚪Leave")],
            [types.KeyboardButton(text="/ai")],
        ]
        return types.ReplyKeyboardMarkup(keyboard=rematch_kb, resize_keyboard=True)
    if room is None:
        return types.ReplyKeyboardRemove()
    return render.human_turn_keyboard(room.engine, room.human_id)


async def _send_events(bot: Bot, chat_id: int, room: "AIRoom", events):
    """Dispatch room-emitted events to Telegram in order.

    A small async sleep between AI actions slows the bot enough that a
    human can read what happened. Not cosmetic-only: without this, a
    multi-item AI turn fires 4-5 messages in the same animation frame
    and the user misses intermediate state.

    Loadout strings are pinned at event-construction time (see
    `AIRoomEvent.loadout_text`), not re-rendered here, because this
    coroutine awaits between sends — the live `room.state` may have
    already advanced to a later reload by the time we announce the
    earlier round.
    """
    final_keyboard = None
    for ev in events:
        kb = _send_keyboard(ev.keyboard_hint, room)
        await bot.send_message(chat_id, ev.text, reply_markup=kb)
        if ev.loadout_text is not None:
            # No `protect_content` — the loadout is just the public
            # round announcement; the earlier flag prevented users from
            # copying or forwarding it without any real purpose.
            await bot.send_message(chat_id, ev.loadout_text)
        final_keyboard = kb
        # Pause slightly between consecutive AI sub-actions so the user
        # has time to read them. Room-emitted events may request a
        # longer pause (e.g. round boundaries); otherwise fall back to
        # the default inter-action beat whenever the wait keyboard is
        # up.
        if ev.pause_after_ms > 0:
            await asyncio.sleep(ev.pause_after_ms / 1000)
        elif ev.keyboard_hint == "wait":
            await asyncio.sleep(1.1)
    return final_keyboard


@router.message(StateFilter(None, GameStates.idle, GameStates.in_ai_game), Command("ai"))
async def start_ai_game(message: Message, bot: Bot, state: FSMContext):
    # Drop any previous AI game cleanly — /ai is also the rematch command.
    _AI_ROOMS.pop(message.chat.id, None)

    try:
        from ai.policy import AIPolicy, AIUnavailable  # noqa: F401
        from ai.room import AIRoom
    except ImportError as exc:
        logger.exception("AI module import failed")
        await message.answer(
            "AI mode requires the `torch` dependency. Install it or ask the "
            f"operator to do so. (error: {exc})",
        )
        return

    try:
        policy = AIPolicy.get()
    except Exception as exc:  # pragma: no cover
        logger.exception("AIPolicy failed to load")
        await message.answer(
            f"Couldn't load the AI: {exc}. Try `/join` for a 2-player game instead.",
        )
        return

    room = AIRoom.new(
        human_name=message.from_user.first_name or "Player",
        policy=policy,
    )
    _AI_ROOMS[message.chat.id] = room
    await state.set_state(GameStates.in_ai_game)

    events = room.start()
    await _send_events(bot, message.chat.id, room, events)


@router.message(
    GameStates.in_ai_game,
    F.text,
    F.text.not_in(["🚪Leave"]),
    F.text.not_contains("/"),
)
async def play_ai_turn(message: Message, bot: Bot, state: FSMContext):
    room = _AI_ROOMS.get(message.chat.id)
    if room is None:
        # Orphaned state (bot restarted mid-game). Clear and prompt.
        await state.clear()
        await message.answer("This game session expired. Use /ai to start a new one.")
        return

    from ai.actions import emoji_to_action

    action = emoji_to_action(
        message.text, adrenaline_active=room.state.adrenaline_active,
    )
    if action is None:
        await message.answer("Make a valid move.")
        return

    events = room.step_human(action)
    await _send_events(bot, message.chat.id, room, events)

    if room.game_over:
        # Drop the room now that the match is done; the user can /ai to
        # rematch. State stays `in_ai_game` so the rematch flow works.
        _AI_ROOMS.pop(message.chat.id, None)


@router.message(GameStates.in_ai_game, Command("leave"))
@router.message(GameStates.in_ai_game, F.text.in_(["🚪Leave"]))
async def leave_ai(message: Message, state: FSMContext):
    _AI_ROOMS.pop(message.chat.id, None)
    await state.clear()
    await message.answer("Game over.", reply_markup=types.ReplyKeyboardRemove())
