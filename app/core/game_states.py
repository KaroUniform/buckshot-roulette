from aiogram.fsm.state import StatesGroup, State


class GameStates(StatesGroup):
    in_game = State()
    new_raund = State()
    idle = State()
