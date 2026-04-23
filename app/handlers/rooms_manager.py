import asyncio
import random
import re
from aiogram import Bot, F, Router, types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.types import Message

from core.game_states import GameStates
from core.room_manager import RoomsManager
from gameplay import render as gameplay_render
from gameplay.session import EngineSession, SessionEvent
from utils.storage import STORAGE


router = Router()
MANAGER = RoomsManager()


async def _place_player_in_waiting_room(message: Message, state: FSMContext, room_id: int):
    MANAGER.create_room(room_id)
    MANAGER.reg_player_in_room(message.from_user.first_name, message.chat.id, room_id)
    await state.set_state(GameStates.in_game)
    await state.set_data({"room_id": room_id})
    await message.answer(
        f"You are in room `{room_id}` now. Waiting for the second player. Send them the code!",
        parse_mode="Markdown",
    )


def _keyboard_for_event(
    session: EngineSession,
    seat: int,
    event: SessionEvent,
    *,
    rematch_room_id: int | None = None,
):
    if event.keyboard_hint == "game_over":
        kb = [[types.KeyboardButton(text="🚪Leave")]]
        if rematch_room_id is not None:
            kb.append([types.KeyboardButton(text=f"/rematch {rematch_room_id}")])
        return types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)
    if event.keyboard_hint == "human_turn":
        return gameplay_render.turn_keyboard(session.engine, seat)
    return gameplay_render.wait_keyboard("🕓Opponent is taking a turn🕓")


async def _send_dispatch(
    bot: Bot,
    session: EngineSession,
    dispatch: dict[int, list[SessionEvent]],
    *,
    rematch_room_id: int | None = None,
):
    max_events = max((len(events) for events in dispatch.values()), default=0)
    for index in range(max_events):
        pause_after_ms = 0
        for chat_id, events in dispatch.items():
            if index >= len(events):
                continue
            seat = session.seat_for_chat(chat_id)
            if seat is None:
                continue
            event = events[index]
            await bot.send_message(
                chat_id,
                event.text,
                reply_markup=_keyboard_for_event(
                    session, seat, event, rematch_room_id=rematch_room_id,
                ),
            )
            pause_after_ms = max(pause_after_ms, event.pause_after_ms)
        if pause_after_ms > 0:
            await asyncio.sleep(pause_after_ms / 1000)


@router.message(Command("find"), StateFilter(None))
async def find_game(message: Message, bot: Bot, state: FSMContext):
    if len(MANAGER.search_lobby) > 0:
        user_id = next(iter(MANAGER.search_lobby))
        user_data = MANAGER.search_lobby.pop(user_id)
        room_id = random.randint(100000, 999999)

        second_user_message = user_data["message"]
        second_user_state = FSMContext(
            storage=STORAGE,
            key=StorageKey(
                chat_id=second_user_message.chat.id,
                user_id=second_user_message.from_user.id,
                bot_id=bot.id,
            ),
        )

        MANAGER.create_room(room_id)
        MANAGER.reg_player_in_room(message.from_user.first_name, message.chat.id, room_id)
        MANAGER.reg_player_in_room(
            second_user_message.from_user.first_name,
            second_user_message.chat.id,
            room_id,
        )

        await state.set_state(GameStates.in_game)
        await state.set_data({"room_id": room_id})
        await second_user_state.set_state(GameStates.in_game)
        await second_user_state.set_data({"room_id": room_id})

        session = MANAGER.start_room_session(room_id)
        dispatch = session.start()
        await _send_dispatch(bot, session, dispatch)
        return

    MANAGER.search_lobby[message.from_user.id] = {"message": message}
    await message.answer(
        "⏳ You are in the waiting lobby. As soon as another player searches for a game, they will join you."
    )
    await state.set_state(GameStates.in_search)


@router.message(Command("leave"))
@router.message(F.text.in_(["🚪Leave"]))
async def leave_game(message: Message, bot: Bot, state: FSMContext):
    if message.from_user.id in MANAGER.search_lobby:
        MANAGER.search_lobby.pop(message.from_user.id, None)
        await state.clear()
        await message.answer("farewell!", reply_markup=types.ReplyKeyboardRemove())
        return

    await state.clear()
    try:
        room_id = MANAGER.get_room_id_by_player(message.chat.id)
    except Exception:
        await message.answer("farewell!", reply_markup=types.ReplyKeyboardRemove())
        return

    room = MANAGER.ROOMS.get(room_id, None)
    player_ids = [player.chat_id for player in room.players] if room else []
    MANAGER.del_player_from_rooms(message.chat.id)

    for player_id in player_ids:
        await bot.send_message(
            player_id,
            f"{message.from_user.first_name} has left the game",
            reply_markup=types.ReplyKeyboardRemove(),
        )


@router.message(Command("rematch"))
@router.message(StateFilter(None), Command("join"))
async def join(message: Message, bot: Bot, state: FSMContext):
    MANAGER.search_lobby.pop(message.from_user.id, None)
    requested_room_id = extract_code_from_string(message.text)
    room_id = requested_room_id

    if room_id is None:
        room_id = random.randint(100000, 999999)
        await _place_player_in_waiting_room(message, state, room_id)
        return

    if not MANAGER.check_room(room_id):
        await _place_player_in_waiting_room(message, state, room_id)
        return

    if not MANAGER.room_can_accept_player(room_id):
        room_id = random.randint(100000, 999999)
        await _place_player_in_waiting_room(message, state, room_id)
        return

    MANAGER.reg_player_in_room(message.from_user.first_name, message.chat.id, room_id)
    await state.set_state(GameStates.in_game)
    await state.set_data({"room_id": room_id})

    session = MANAGER.start_room_session(room_id)
    dispatch = session.start()
    await _send_dispatch(bot, session, dispatch)


@router.message(
    GameStates.in_game,
    F.text,
    F.text.not_in(["🚪Leave"]),
    F.text.not_contains("/"),
)
async def in_game(message: Message, bot: Bot, state: FSMContext):
    try:
        session = MANAGER.get_session_by_player(message.chat.id)
    except Exception:
        await state.clear()
        return

    dispatch = session.handle_text(message.chat.id, message.text)
    rematch_room_id = random.randint(100000, 999999) if session.game_over else None
    await _send_dispatch(bot, session, dispatch, rematch_room_id=rematch_room_id)

    if session.game_over:
        MANAGER.del_player_from_rooms(message.chat.id)


def extract_code_from_string(input_string):
    pattern = r"/\w+ (\d{6})"
    match = re.search(pattern, input_string)

    if match:
        code = match.group(1)
        return int(code)
    return None
