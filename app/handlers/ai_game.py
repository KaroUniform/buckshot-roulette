"""Telegram handler for the single-player AI mode."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict

from aiogram import Bot, F, Router, types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from ai.policy import AIUnavailable, AIPolicy
from core.game_states import GameStates
from gameplay import render as gameplay_render
from gameplay.session import EngineSession, SessionEvent
from handlers.winner_sticker import send_winner_sticker

logger = logging.getLogger(__name__)

router = Router()

_AI_ROOMS: Dict[int, EngineSession] = {}
DEFAULT_AI_WAIT_PAUSE_MS = 1100


def _keyboard_for_event(session: EngineSession, seat: int, event: SessionEvent):
    if event.keyboard_hint == "game_over":
        rematch_kb = [
            [types.KeyboardButton(text="🚪Leave")],
            [types.KeyboardButton(text="/ai")],
        ]
        return types.ReplyKeyboardMarkup(keyboard=rematch_kb, resize_keyboard=True)
    if event.keyboard_hint == "human_turn":
        return gameplay_render.turn_keyboard(session.engine, seat)
    return gameplay_render.wait_keyboard("🕓AI is thinking🕓")


async def _send_events(bot: Bot, chat_id: int, session: EngineSession, events: list[SessionEvent]):
    seat = session.seat_for_chat(chat_id)
    if seat is None:
        return

    sticker_sent = False
    for event in events:
        if (
            not sticker_sent
            and event.event_type == "game_over"
            and session.ai_won()
        ):
            await send_winner_sticker(bot, chat_id)
            sticker_sent = True

        await bot.send_message(
            chat_id,
            event.text,
            reply_markup=_keyboard_for_event(session, seat, event),
        )
        if event.pause_after_ms > 0:
            await asyncio.sleep(event.pause_after_ms / 1000)
        elif event.keyboard_hint == "wait":
            await asyncio.sleep(DEFAULT_AI_WAIT_PAUSE_MS / 1000)


@router.message(StateFilter(None, GameStates.idle, GameStates.in_ai_game), Command("ai"))
async def start_ai_game(message: Message, bot: Bot, state: FSMContext):
    _AI_ROOMS.pop(message.chat.id, None)

    try:
        policy = AIPolicy.get()
    except AIUnavailable as exc:
        logger.exception("AI policy unavailable")
        await message.answer(
            "AI mode requires the `torch` dependency and the checkpoint. "
            f"(error: {exc})",
        )
        return
    except Exception as exc:  # pragma: no cover
        logger.exception("AIPolicy failed to load")
        await message.answer(
            f"Couldn't load the AI: {exc}. Try `/join` for a 2-player game instead.",
        )
        return

    session = EngineSession.new_vs_ai(
        human_name=message.from_user.first_name or "Player",
        human_chat_id=message.chat.id,
        policy=policy,
    )
    _AI_ROOMS[message.chat.id] = session
    await state.set_state(GameStates.in_ai_game)

    dispatch = session.start()
    await _send_events(bot, message.chat.id, session, dispatch.get(message.chat.id, []))
    await _record_if_ended(message)


@router.message(
    GameStates.in_ai_game,
    F.text,
    F.text.not_in(["🚪Leave"]),
    F.text.not_contains("/"),
)
async def play_ai_turn(message: Message, bot: Bot, state: FSMContext):
    session = _AI_ROOMS.get(message.chat.id)
    if session is None:
        await state.clear()
        await message.answer("This game session expired. Use /ai to start a new one.")
        return

    dispatch = session.handle_text(message.chat.id, message.text)
    await _send_events(bot, message.chat.id, session, dispatch.get(message.chat.id, []))

    await _record_if_ended(message)


async def _record_if_ended(message: Message) -> None:
    """If the room is terminal, persist the match and drop the room.

    Safe to call on non-terminal states — becomes a no-op. Record first,
    pop second, so a stats-store crash doesn't leave the room orphaned
    with no way to retry the record.
    """
    session = _AI_ROOMS.get(message.chat.id)
    if session is None or not session.game_over:
        return
    # Guard the stats write behind try/except so a transient DB issue
    # (disk full, locked file) can't poison the user's game-over UX —
    # they just don't see this match in /stats. Log the failure loudly.
    try:
        from ai.stats import MatchRecord, get_stats

        store = await get_stats()
        ended_at = datetime.now(timezone.utc)
        started = session.started_at or ended_at
        duration_ms = int((ended_at - started).total_seconds() * 1000)
        await store.record(MatchRecord(
            ended_at=ended_at,
            chat_id=message.chat.id,
            user_id=(message.from_user.id if message.from_user else None),
            human_name=session.names[0],
            ai_won=session.ai_won(),
            human_went_first=session.human_went_first,
            n_human_turns=session.n_human_turns,
            n_ai_actions=session.n_ai_actions,
            n_reloads=int(session.state.n_reloads),
            seed=session.seed,
            duration_ms=max(0, duration_ms),
        ))
    except Exception:  # pragma: no cover — defensive
        logger.exception("failed to record match stats")
    finally:
        # Always drop the room — the match is over either way. State
        # stays `in_ai_game` so /ai rematch flow works.
        _AI_ROOMS.pop(message.chat.id, None)


@router.message(GameStates.in_ai_game, Command("leave"))
@router.message(GameStates.in_ai_game, F.text.in_(["🚪Leave"]))
async def leave_ai(message: Message, state: FSMContext):
    _AI_ROOMS.pop(message.chat.id, None)
    await state.clear()
    await message.answer("Game over.", reply_markup=types.ReplyKeyboardRemove())
