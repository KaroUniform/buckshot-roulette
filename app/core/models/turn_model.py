from core.models.items import ItemType
from pydantic import BaseModel
from typing import Literal, Optional, List
from copy import deepcopy
import random


class TurnResult(BaseModel):
    """
    Represents the result of a turn in the game.

    Attributes:
    - first_player_hp (Optional[str]): Health points information for the first player.
    - second_player_hp (Optional[str]): Health points information for the second player.
    - first_player_items (Optional[List]): Items information for the first player.
    - second_player_items (Optional[List]): Items information for the second player.
    - rounds (Optional[str]): Shotgun rounds information.
    - active_player_action_result (Optional[str]): Result of the action performed by the active player.
    - passive_player_action_result (Optional[str]): Result of the action perceived by the passive player.
    - is_game_ended (bool): Indicates if the game has ended.
    - give_turn (Optional[bool]): Indicates whether the turn is being given to the other player.
    - on_start_first_id (Optional[int]): The ID of the player starting the game.
    - next_turn (Optional[int]): The index of the next turn in the game.

    Usage:
    Create an instance of TurnResult to store and communicate the results of a game turn.
    """

    first_player_hp: Optional[str] = None
    second_player_hp: Optional[str] = None
    first_player_items: Optional[List] = []
    second_player_items: Optional[List] = []
    rounds: Optional[str] = None
    active_player_action_result: Optional[str] = None
    passive_player_action_result: Optional[str] = None
    is_game_ended: bool = False
    give_turn: Optional[bool] = None
    on_start_first_id: Optional[int] = None
    next_turn: Optional[int] = None


class Dealer:
    """
    Manages the turns and actions in the game.

    Attributes:
    - max_items_per_turn (int): The maximum number of items allowed to be used per turn.
    - first_player (int): The ID of the first player.
    - second_player (int): The ID of the second player.
    - move_counter (int): Counter to keep track of the current turn.
    - queue (List[Turn]): List of turns in the game.

    Methods:
    - end(result: str): Ends the current turn by assigning the result.
    - extend_turn(): Extends the current turn by creating a new turn with cleared used items.
    - use_items(item: str) -> bool: Records the usage of an item during the turn.
    - turn_int() -> int: Returns the turn number.
    - player_id() -> int: Returns the player ID of the current turn.
    """

    def __init__(
        self, first_player: int, second_player: int, max_items_per_turn=1
    ) -> None:
        """
        Initializes the Dealer with player IDs and maximum items allowed per turn.
        """
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
        """
        Ends the current turn by assigning the result.

        Args:
        - result (str): The result of the current turn.
        """
        turn = self.queue[self.move_counter]
        turn.turn_result = result
        self.move_counter += 1

    def extend_turn(self):
        """
        Extends the current turn by creating a new turn with cleared used items.
        """
        turn = self.queue[self.move_counter]
        new_turn = deepcopy(turn)
        new_turn.used_items = []
        self.queue.insert(self.move_counter + 1, new_turn)

    def use_items(self, item: ItemType) -> bool:
        """
        Records the usage of an item during the turn.

        Args:
        - item (str): The item to be used.

        Returns:
        - bool: True if the item usage is successful, False otherwise.
        """
        items = self.queue[self.move_counter].used_items
        if sum(1 for item in items if item != ItemType.ADRENALINE) > 0:
            return False
        items.append(item)
        return True

    @property
    def turn_int(self) -> int:
        """
        Returns the turn number.

        Returns:
        - int: The current turn number.
        """
        return self.queue[self.move_counter].turn_number

    @property
    def player_id(self) -> int:
        """
        Returns the player ID of the current turn.

        Returns:
        - int: The player ID of the current turn.
        """
        return self.queue[self.move_counter].player_id


class Turn:
    """
    Represents a single turn in the game.

    Attributes:
    - turn_number (Literal[1, 0]): The turn number, indicating the active player.
    - player_id (int): The ID of the player for the turn.
    - used_items (List[str]): List of items used during the turn.
    - turn_result (str): Result of the turn.

    Usage:
    Create an instance of Turn to represent a single turn in the game.
    """

    def __init__(self, turn_number: Literal[1, 0], player_id: int) -> None:
        """
        Initializes a Turn with a turn number and player ID.
        """
        self.player_id = player_id
        self.turn_number = turn_number
        self.used_items = []
        self.turn_result = ""
