import random
from pydantic import BaseModel
from typing import Literal, Optional
from copy import deepcopy


class TurnResult(BaseModel):

    first_player_hp: Optional[str] = None
    second_player_hp: Optional[str] = None
    first_player_items: Optional[list] = []
    second_player_items: Optional[list] = []
    rounds: Optional[str] = None
    active_player_action_result: Optional[str] = None
    passive_player_action_result: Optional[str] = None
    is_game_ended: bool = False
    give_turn: Optional[bool] = None
    on_start_first_id: Optional[int] = None
    next_turn: Optional[int] = None


class Dealer:

    def __init__(
        self, first_player: int, second_player: int, max_items_per_turn=1
    ) -> None:
        self.max_items_per_turn = max_items_per_turn
        self.first_player = first_player
        self.second_player = second_player
        self.move_counter = 0
        list2 = [Turn(1, second_player) for _ in range(100)]
        list1 = [Turn(0, first_player) for _ in range(100)]
        if random.choice([0, 1]) == 0:
            self.queue = [item for pair in zip(list1, list2) for item in pair]
        else:
            self.queue = [item for pair in zip(list2, list1) for item in pair]

    def end(self, result: str):
        turn = self.queue[self.move_counter]
        turn.turn_result = result
        self.move_counter += 1

    def extend_turn(self):
        turn = self.queue[self.move_counter]
        new_turn = deepcopy(turn)
        new_turn.used_items = []
        self.queue.insert(self.move_counter + 1, new_turn)

    def use_items(self, item: str):
        items = self.queue[self.move_counter].used_items
        if len(items) > 0:
            return False
        items.append(item)
        return True

    @property
    def turn_int(self):
        return self.queue[self.move_counter].turn_number

    @property
    def player_id(self):
        return self.queue[self.move_counter].player_id


class Turn:

    def __init__(self, turn_number: Literal[1, 0], player_id: int) -> None:
        self.player_id = player_id
        self.turn_number = turn_number
        self.used_items = []
        self.turn_result = ""
