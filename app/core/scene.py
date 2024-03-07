from typing import List, Literal, Tuple, Union


from core.models.turn_model import TurnResult, Dealer
from .models.player_model import PlayerModel
from .player import Player
from .shotgun import Shotgun
import random


from core.models.items import ItemType
from core.models.ammo import AmmoType


class Scene:
    """
    Represents the game scene and manages the game logic.

    Attributes:
    - first_player (Player): The first player in the game.
    - second_player (Player): The second player in the game.
    - shotgun (Shotgun): The shotgun used in the game.
    - ITEMS (list): List of available item types for distribution.
    - games (int): Number of games to play. #WIP
    - game_ended (bool): Indicates whether the current game has ended.
    - score (int): Current score of the game. #WIP
    """

    first_player: Player
    second_player: Player
    shotgun: Shotgun
    games: int
    game_ended: bool
    score = 0

    # Manage the probability of items spawn by it's counts
    ITEMS = [
        ItemType.HANDSAW,
        ItemType.BEER,
        ItemType.BEER,
        ItemType.BEER,
        ItemType.HANDCUFF,
        ItemType.HANDCUFF,
        ItemType.SMOKE,
        ItemType.SMOKE,
        ItemType.GLASS,
    ]

    def __init__(self, games: int = 3) -> None:
        self.games = games
        self.shotgun = Shotgun()
        self.game_ended = True
        self.first_player = None
        self.second_player = None

    def add_player(self, model: PlayerModel, player_seat: int):
        """
        Adds a player to the game.

        Args:
        - model (PlayerModel): The player model.
        - player_seat (int): The seat of the player (0 or 1).
        """
        if player_seat == 0:
            self.first_player = Player(model)
        else:
            self.second_player = Player(model)

    def start(self) -> TurnResult:
        """
        Init a new game. Call it once when game is starting

        Returns:
        - TurnResult: The result of the initial game setup.
        """

        # Create game logic manager to control the game
        self.dealer = Dealer(
            self.first_player.data.chat_id, self.second_player.data.chat_id
        )
        turn_result = TurnResult()

        # Set random HP for players
        hp = random.randint(3, 6)
        self.first_player.set_hp(hp)
        self.second_player.set_hp(hp)

        # Flush player inventories
        self.first_player.flush_inventory()
        self.second_player.flush_inventory()

        # Update the pipeline of moves

        # Determine starting player and set turn result
        starting_player = (
            self.first_player if self.dealer.turn_int == 0 else self.second_player
        )
        _res = f"🪙{starting_player.data.name} is going first"
        turn_result.on_start_first_id = starting_player.data.chat_id

        # Set turn result properties
        turn_result.give_turn = False
        turn_result.active_player_action_result = _res
        turn_result.passive_player_action_result = _res

        # Recharge shotgun
        self.shotgun.recharge()

        # Set a number of items to distribute
        n = random.choice([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4])

        # Distribute items to players
        self.distribute_items(self.first_player, n)
        self.distribute_items(self.second_player, n)

        # Add HP information to turn result
        self.add_hp_info(turn_result)

        # Set player items and shotgun rounds in turn result
        turn_result.first_player_items = self.first_player.get_items_emoji()
        turn_result.second_player_items = self.second_player.get_items_emoji()
        turn_result.rounds = self.shotgun.get_ammo_sorted()

        # Set game state
        self.game_ended = False
        turn_result.next_turn = self.dealer.turn_int

        return turn_result

    def make_turn(
        self,
        action: Literal[
            "handsaw", "beer", "smoke", "handcuff", "magnifying_glass", "me", "him"
        ],
        player_id: int,
    ):
        """
        Performs a player's turn in the game.

        - action (Literal): The action to perform. Valid actions include:
            - "handsaw": Use a handsaw to deal damage.
            - "beer": Consume beer for a chance to affect shotgun rounds.
            - "smoke": Use smoke to increase player's health.
            - "handcuff": Use handcuffs to immobilize the opponent temporarily.
            - "magnifying_glass": Inspect the shotgun with a magnifying glass.
            - "me": Player chooses to perform an action on themselves.
            - "him": Player chooses to perform an action on the opponent.
        - player_id (int): The ID of the player making the turn. The dealer will not allow the player to make a wrong turn.

        Returns:
        - TurnResult: The dataclass of result of the player's turn.
        """

        turn_con = self.dealer.turn_int  # Get the current turn index
        turn_result = TurnResult()  # Create a TurnResult object to store the result

        # Check if it's the correct player's turn
        if self.dealer.player_id != player_id:
            turn_result.active_player_action_result = "Please, wait your turn"
            return turn_result

        # Determine the active and passive players for the current turn
        player, target = self.solve_players()

        # Process the chosen action based on the provided action string
        if action == "me":
            # Handle action when player chooses to perform an action on themselves
            action_results = self.shotgun.shot(target=player)
            if action_results == AmmoType.LIVE.value:
                self.first_player.still_tied()
                self.second_player.still_tied()
            else:
                self.dealer.extend_turn()
            turn_result.active_player_action_result = action_results
            turn_result.passive_player_action_result = (
                f"{player.data.name} {action_results} itself!"
            )
            self.dealer.end(action_results)

        elif action == "him":
            # Handle action when player chooses to perform an action on the opponent
            action_results = self.shotgun.shot(target=target)
            self.dealer.end(action_results)
            turn_result.active_player_action_result = f"You shot by {action_results}"
            turn_result.passive_player_action_result = (
                f"{player.data.name} {action_results} to you"
            )
            self.first_player.still_tied()
            self.second_player.still_tied()

        else:
            # Handle action when player uses an item
            action_results, second_player_action_result = self.use_item(action)
            turn_result.active_player_action_result = action_results
            turn_result.passive_player_action_result = second_player_action_result

        # Check game state and update TurnResult accordingly
        if not self.first_player.still_alive():
            self.handle_game_end(self.second_player, turn_result)

        elif not self.second_player.still_alive():
            self.handle_game_end(self.first_player, turn_result)

        elif self.shotgun.need_recharge():
            self.handle_recharge(turn_result, turn_con)
        else:
            self.handle_next_player(turn_result, turn_con)

        turn_result.next_turn = self.dealer.turn_int
        return turn_result

    def handle_game_end(self, winner_player: Player, turn_result: TurnResult):
        """
        Handles the conclusion of a game when a player wins.

        Args:
        - winner_player (Player): The player who has won the game.
        - turn_result (TurnResult): The TurnResult object to update with the game end information.
        """
        # Update the game score based on the winner
        self.score += 1 if winner_player == self.first_player else -1

        # Set the game as ended
        self.game_ended = True

        # Prepare the action results message
        action_results = f"{winner_player.data.name} has won!\n"
        turn_result.active_player_action_result = action_results
        turn_result.passive_player_action_result = action_results

        # Indicate that the game has ended in the TurnResult
        turn_result.is_game_ended = True

        # Update player items in TurnResult
        turn_result.first_player_items = self.first_player.get_items_emoji()
        turn_result.second_player_items = self.second_player.get_items_emoji()

    def handle_recharge(self, turn_result: TurnResult, turn_con):
        """
        Handles the recharge phase of the game, including shotgun reloading and item distribution.

        Args:
        - turn_result (TurnResult): The TurnResult object to update with recharge information.
        - turn_con: The current turn index for reference.

        Note:
        This method assumes that the shotgun needs to be recharged.
        """
        # Reload shotgun rounds and get the updated rounds in TurnResult
        turn_result.rounds = self.shotgun.recharge()

        # Determine the number of items to distribute randomly
        n = random.choice([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])

        # Distribute items to the first player and update TurnResult
        blackbox_one = self.distribute_items(self.first_player, n)
        turn_result.first_player_items = blackbox_one

        # Distribute items to the second player and update TurnResult
        blackbox_two = self.distribute_items(self.second_player, n)
        turn_result.second_player_items = blackbox_two

        # Reset tied status for both players
        self.first_player.tied = 0
        self.second_player.tied = 0

        # Add next player message and update HP information in TurnResult
        self.add_next_player_message(turn_result, turn_con)
        self.add_hp_info(turn_result)

    def handle_next_player(self, turn_result: TurnResult, turn_con):
        """
        Prepares the game state for the next player's turn, updating TurnResult accordingly.

        Args:
        - turn_result (TurnResult): The TurnResult object to update with the next player information.
        - turn_con: The current turn index for reference.
        """
        # Add next player message and update HP information in TurnResult
        self.add_next_player_message(turn_result, turn_con)
        self.add_hp_info(turn_result)

        # Update player items in TurnResult
        turn_result.first_player_items = self.first_player.get_items_emoji()
        turn_result.second_player_items = self.second_player.get_items_emoji()

        # Set shotgun rounds in TurnResult
        turn_result.rounds = self.shotgun.get_ammo_sorted()

    def add_hp_info(self, turn_result: TurnResult):
        """
        Adds information about players' health points (HP) to the TurnResult.

        Args:
        - turn_result (TurnResult): The TurnResult object to update with HP information.
        """
        # Add first player's HP information to TurnResult
        turn_result.first_player_hp = (
            f"{self.first_player.data.name} : {'⚡️'*self.first_player.data.hp}\n"
        )

        # Add second player's HP information to TurnResult
        turn_result.second_player_hp = (
            f"{self.second_player.data.name} : {'⚡️'*self.second_player.data.hp}\n"
        )

    # TODO Rewrite logic
    def add_next_player_message(self, turn_result: TurnResult, turn_con: int):
        if self.dealer.turn_int != turn_con:
            turn_result.give_turn = True

    def use_item(
        self,
        item: Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"],
    ):
        """
        Processes a player's use of a specific item during their turn.

        Args:
        - item (Literal): The item to use. Valid items include:
            - "handsaw": Use a handsaw to double damage.
            - "beer": Consume beer to throw away the last round from the shotgun.
            - "smoke": Use smoke to increase player's health.
            - "handcuff": Use handcuffs to immobilize the opponent temporarily.
            - "magnifying_glass": Inspect the shotgun with a magnifying glass.

        Returns:
        - Tuple[Union[str, Tuple[str, str]], str]:
            - If the item usage is successful, a tuple containing:
                - The result of the item usage for the active player.
                - The result of the item usage for the passive player.
            - If the item usage fails, a string indicating the failure reason.
        """
        player, target = self.solve_players()

        if player.get_number_of_item(item) == 0:
            return (f"You don't have {item}", None)

        if not self.dealer.use_items(item):
            return "❌Only 1 item per turn", None

        match item:
            case ItemType.HANDSAW.value:

                player.inventory.handsaw -= 1
                self.shotgun.damage *= 2
                return (
                    "🧨Careful, the weapon now deals x2 damage",
                    f"{player.data.name} uses🪚",
                )
            case ItemType.BEER.value:

                player.inventory.beer -= 1
                round = self.shotgun.shaking()
                return (
                    f"⤴️ The {round} flew out of the shotgun",
                    f"{player.data.name} uses 🍺. The ⤴️{round} flew out of the shotgun ",
                )
            case ItemType.SMOKE.value:

                player.inventory.smoke -= 1
                player.smoke()
                return (
                    f"🚬You now have {player.data.hp} hp",
                    f"{player.data.name} uses 🚬",
                )
            case ItemType.HANDCUFF.value:

                if target.tied > 0:
                    return ("You cannot re-link a linked player", None)
                player.inventory.handcuff -= 1
                target.tied = 3
                self.dealer.extend_turn()
                return (
                    f"🔗{target.data.name} is tied for 1 turn",
                    f"🔗 {player.data.name} has tied you up",
                )
            case ItemType.GLASS.value:

                player.inventory.magnifying_glass -= 1
                return (
                    f"🔍 You see the {str(self.shotgun.inspect())} inside",
                    f"{player.data.name} uses 🔍. Very interesting...",
                )

    def distribute_items(self, player: Player, desired_length: int) -> List[str]:
        """
        Distributes a specified number of random items to a player.

        Args:
        - player (Player): The player to receive the distributed items.
        - desired_length (int): The desired number of items to distribute.

        Returns:
        - List[str]: A list containing emoji representations of the distributed items.

        Note:
        If the inventory is overflowing, the distribution stops.
        """
        # Create a list of randomly selected items for distribution
        blackbox = [random.choice(self.ITEMS) for _ in range(desired_length)]

        # Distribute the items to the player, updating their inventory
        for item in blackbox:
            try:
                player.add_item(item.value)
            except Exception:
                break

        # Return emoji representations of the distributed items
        return player.get_items_emoji()

    def get_current_username(self) -> str:
        """
        Retrieves the name of the current active player.

        Returns:
        - str: The username of the current active player.
        """
        # Determine the current active player
        player, _ = self.solve_players()

        # Return the username of the current active player
        return player.data.name

    def solve_players(self) -> Tuple[Player, Player]:
        """
        Determines the active and passive players based on the current turn.

        Returns:
        - Tuple[Player, Player]: A tuple containing the active player and passive player.
        """
        # Check if the turn index is even or odd to determine active and passive players
        if self.dealer.turn_int % 2 == 0:
            player = self.first_player
            target = self.second_player
        else:
            target = self.first_player
            player = self.second_player

        # Return the active and passive players
        return player, target
