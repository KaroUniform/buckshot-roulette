from aiogram.fsm.state import StatesGroup, State


class GameStates(StatesGroup):
    in_game = State()
    in_search = State()
    new_raund = State()
    idle = State()
    # Player is fighting the E19 policy in single-player mode. Kept
    # separate from `in_game` so the multiplayer message router can't
    # accidentally claim an AI-mode message (and vice-versa).
    in_ai_game = State()
