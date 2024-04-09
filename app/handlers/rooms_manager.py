import random
import re
from aiogram import F, Bot, Router, types
from aiogram.filters import Command
from aiogram.types import Message
from core.models.items import ItemType
from core.player import Player
from core.models.turn_model import TurnResult
from core.room_manager import RoomsManager
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from utils.storage import STORAGE

from core.game_states import GameStates
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.filters import StateFilter
import asyncio
from utils.edit_message_with_delay import edit_message



router = Router()
MANAGER = RoomsManager()


@router.message(Command("find"), StateFilter(None))
async def find_game(message: Message, bot: Bot, state: FSMContext):
    
    if len(MANAGER.search_lobby) > 0:
        user_id = next(iter(MANAGER.search_lobby))
        user_data = MANAGER.search_lobby.pop(user_id)
        room_id = random.randint(100000, 999999)
        
        message = message.model_copy(update={"text": "/join " + str(room_id)})
        await join(message=message, bot=bot, state=state)
        
        second_user_message = user_data['message']
        second_user_message = second_user_message.model_copy(update={"text": "/join " + str(room_id)})
        
        second_user_state = FSMContext(
            storage=STORAGE,
            key=StorageKey(
                chat_id=second_user_message.chat.id,
                user_id=second_user_message.from_user.id,  
                bot_id=bot.id
            )
        )
        
        await join(message=second_user_message, bot=bot, state=second_user_state)
        
    else:
        MANAGER.search_lobby[message.from_user.id] = {'message': message}
        await message.answer("⏳You are in the waiting lobby. As soon as another player tries to find a game, he will join you.")
        await state.set_state(GameStates.in_search)
    

@router.message(Command("leave"))
@router.message(F.text.in_(["🚪Leave"]))
async def new_round(message: Message, bot: Bot, state: FSMContext):
    if state == GameStates.in_search:
        #TODO
        pass
    
    await state.clear()
    try:
        first_player, second_player = MANAGER.get_players_chatid(message.chat.id)
    except Exception as e:
        await message.answer("farewell!", reply_markup=types.ReplyKeyboardRemove())
        return

    MANAGER.del_player_from_rooms(first_player)
    MANAGER.del_player_from_rooms(second_player)

    if first_player is not None:
        await bot.send_message(
            first_player,
            f"{message.from_user.first_name} has left the game",
            reply_markup=types.ReplyKeyboardRemove(),
        )
    if second_player is not None:
        await bot.send_message(
            second_player,
            f"{message.from_user.first_name} has left the game",
            reply_markup=types.ReplyKeyboardRemove(),
        )


@router.message(
    Command("rematch"),
)
@router.message(
    StateFilter(None),
    Command("join"),
)
async def join(message: Message, bot: Bot, state: FSMContext):

    room_id = extract_code_from_string(message.text)

    if not room_id or not MANAGER.check_room(room_id):
        room_id = room_id or random.randint(100000, 999999)
        MANAGER.create_room(room_id)
        MANAGER.reg_player_in_room(
            message.from_user.first_name, message.chat.id, room_id
        )
        await state.set_state(GameStates.in_game)
        await state.set_data({"room": room_id})
        await message.answer(
            f"You are in the room `{room_id}` now. Waiting to second player. Send him the code!",
            parse_mode="Markdown",
        )
        return

    room = MANAGER.get_room(room_id)
    MANAGER.reg_player_in_room(message.from_user.first_name, message.chat.id, room_id)
    await state.set_state(GameStates.in_game)
    await state.set_data({"room_id": room_id})
    first_player, second_player = MANAGER.get_players_chatid(message.chat.id)

    turn_result = room.start()
    first_player, second_player = MANAGER.get_players_chatid(
        turn_result.on_start_first_id
    )
    passive_user_id = (
        second_player if first_player == turn_result.on_start_first_id else first_player
    )
    await send_game_messages(
        bot,
        turn_result,
        active_user_id=turn_result.on_start_first_id,
        passive_user_id=passive_user_id,
        need_update=True,
        loadout_show_time=101
    )

async def send_info(bot: Bot, user_id: int, turn_result: TurnResult):
    res = (
        turn_result.first_player_hp
        + "Items: "
        + ",".join(turn_result.first_player_items)
        + "\n"
        + turn_result.second_player_hp
        + "Items: "
        + ",".join(turn_result.second_player_items)
    )
    await bot.send_message(user_id, res)


async def send_game_messages(
    bot: Bot,
    turn_result: TurnResult,
    active_user_id: int,
    passive_user_id: int,
    need_update: bool,
    loadout_show_time = 4
):

    active_user_items = (
        turn_result.first_player_items
        if turn_result.next_turn == 0
        else turn_result.second_player_items
    )
    passive_user_items = (
        turn_result.second_player_items
        if turn_result.next_turn == 1
        else turn_result.first_player_items
    )

    builder = ReplyKeyboardBuilder()
    if not turn_result.give_turn:
        builder.row(types.KeyboardButton(text="🔼"))
        first_player_items = [
            types.KeyboardButton(text=item) for item in active_user_items
        ]
        builder.row(*first_player_items)
        builder.row(types.KeyboardButton(text="🔽"))
    else:
        builder.row(types.KeyboardButton(text="🕓Please, wait🕓"))
    await bot.send_message(
        active_user_id,
        turn_result.active_player_action_result,
        reply_markup=builder.as_markup(resize_keyboard=True),
    )
    await send_info(bot, active_user_id, turn_result)
    
    if(turn_result.rounds):
        msg1 = await bot.send_message(active_user_id, turn_result.rounds, protect_content=True)
        msg2 = await bot.send_message(passive_user_id, turn_result.rounds, protect_content=True)
        asyncio.create_task(edit_message(msg1, sleep=loadout_show_time))
        asyncio.create_task(edit_message(msg2, sleep=loadout_show_time))

    if not turn_result.passive_player_action_result:
        return

    builder = ReplyKeyboardBuilder()
    if turn_result.give_turn:
        builder.row(types.KeyboardButton(text="🔼"))
        second_player_items = [
            types.KeyboardButton(text=item) for item in passive_user_items
        ]
        builder.row(*second_player_items)
        builder.row(types.KeyboardButton(text="🔽"))
    else:
        builder.row(types.KeyboardButton(text="🕓Please, wait🕓"))
    await bot.send_message(
        passive_user_id,
        turn_result.passive_player_action_result,
        reply_markup=builder.as_markup(resize_keyboard=True),
    )
    if turn_result.give_turn or need_update:
        await send_info(bot, passive_user_id, turn_result)


@router.message(GameStates.in_game, F.text, F.text.not_in(["🚪Leave"]), F.text.not_contains("/"))
async def in_game(message: Message, bot: Bot, state: FSMContext):
    try:
        room_id = MANAGER.get_room_id_by_player(message.chat.id)
    except Exception:
        await state.clear()
        return

    room = MANAGER.get_room(room_id)
    active_user, passive_user = room.solve_players()

    if message.chat.id != active_user.data.chat_id:
        await message.answer("Please, wait your turn")
        return

    need_update = False

    action_mapping = {
        "🔼": "him",
        "🔽": "me",
        "🍺": ItemType.BEER,
        "🚬": ItemType.SMOKE,
        "🪚": ItemType.HANDSAW,
        "🔗": ItemType.HANDCUFF,
        "🔍": ItemType.GLASS,
        "💉": ItemType.ADRENALINE,
        "🔀": ItemType.INVERTER,
        "📞": ItemType.PHONE,
        "💊": ItemType.PILLS,
    }

    action = action_mapping.get(message.text[0], None)

    if action is None:
        await message.answer("Make a valid turn")
        return

    turn_result = room.make_turn(action, message.chat.id)

    if turn_result.is_game_ended:
        room_id = random.randint(100000, 999999)
        kb = [
            [types.KeyboardButton(text="🚪Leave")],
            [types.KeyboardButton(text=f"/rematch {room_id}")],
        ]
        keyboard_die = types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)
        keyboard_win = types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

        if passive_user.data.hp < 1:
            await send_game_end_message(
                bot, room, keyboard_die, keyboard_win, passive_user, active_user
            )
        else:
            await send_game_end_message(
                bot, room, keyboard_die, keyboard_win, active_user, passive_user
            )

    else:
        await send_game_messages(
            bot,
            turn_result,
            active_user.data.chat_id,
            passive_user.data.chat_id,
            need_update,
        )


async def send_game_end_message(
    bot: Bot, room, keyboard_die, keyboard_win, loser: Player, winner: Player
):
    await bot.send_message(loser.data.chat_id, "⚰️You died", reply_markup=keyboard_die)
    await bot.send_message(
        winner.data.chat_id, "💼Congratulations, you've won!", reply_markup=keyboard_win
    )
    MANAGER.del_player_from_rooms(loser.data.chat_id)
    MANAGER.del_player_from_rooms(winner.data.chat_id)


def extract_code_from_string(input_string):
    pattern = r"/\w+ (\d{6})"
    match = re.search(pattern, input_string)

    if match:
        code = match.group(1)
        return int(code)
    else:
        return None
