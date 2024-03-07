from typing import Literal, Union


from core.models.turn_model import TurnResult, Dealer
from .models.player_model import PlayerModel
from .player import Player
from .shotgun import Shotgun
import random


from core.models.items import ItemType
from core.models.ammo import AmmoType


class Scene:

    first_player: Player
    second_player: Player
    shotgun: Shotgun

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

    games: int
    game_ended: bool
    score = 0

    def __init__(self, games: int = 3) -> None:
        self.games = games
        self.shotgun = Shotgun()
        self.game_ended = True
        self.first_player = None
        self.second_player = None

    def add_player(self, model: PlayerModel, player_turn: int):
        if player_turn == 0:
            self.first_player = Player(model)
        else:
            self.second_player = Player(model)

    # def update_pipe(self):
    # self.pipe = [1, 0] * 50 if random.choice([0, 1]) == 0 else [0, 1] * 50

    def start(self):
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

        turn_con = self.dealer.turn_int
        turn_result = TurnResult()
        if self.dealer.player_id != player_id:
            turn_result.active_player_action_result = "Please, wait your turn"
            return turn_result
        player, target = self.solve_players()

        if action == "me":
            action_results = self.shotgun.shot(player=player)
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
            action_results = self.shotgun.shot(player=target)
            self.dealer.end(action_results)
            turn_result.active_player_action_result = f"You shot by {action_results}"
            turn_result.passive_player_action_result = (
                f"{player.data.name} {action_results} to you"
            )
            self.first_player.still_tied()
            self.second_player.still_tied()

        else:
            action_results, second_player_action_result = self.use_item(action)
            turn_result.active_player_action_result = action_results
            turn_result.passive_player_action_result = second_player_action_result

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
        self.score += 1 if winner_player == self.first_player else -1
        self.game_ended = True
        action_results = f"{winner_player.data.name} has won!\n"
        turn_result.active_player_action_result = action_results
        turn_result.passive_player_action_result = action_results
        turn_result.is_game_ended = True
        turn_result.first_player_items = self.first_player.get_items_emoji()
        turn_result.second_player_items = self.second_player.get_items_emoji()

    def handle_recharge(self, turn_result: TurnResult, turn_con):
        turn_result.rounds = self.shotgun.recharge()
        n = random.choice([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
        blackbox_one = self.distribute_items(self.first_player, n)
        blackbox_two = self.distribute_items(self.second_player, n)
        self.first_player.tied = 0
        self.second_player.tied = 0
        turn_result.first_player_items = blackbox_one
        turn_result.second_player_items = blackbox_two
        self.add_next_player_message(turn_result, turn_con)
        self.add_hp_info(turn_result)

    def handle_next_player(self, turn_result: TurnResult, turn_con):
        self.add_next_player_message(
            turn_result,
            turn_con,
        )
        self.add_hp_info(turn_result)
        turn_result.first_player_items = self.first_player.get_items_emoji()
        turn_result.second_player_items = self.second_player.get_items_emoji()
        turn_result.rounds = self.shotgun.get_ammo_sorted()

    def add_hp_info(self, turn_result: TurnResult):
        turn_result.first_player_hp = (
            f"{self.first_player.data.name} : {'⚡️'*self.first_player.data.hp}\n"
        )
        turn_result.second_player_hp = (
            f"{self.second_player.data.name} : {'⚡️'*self.second_player.data.hp}\n"
        )

    def add_next_player_message(self, turn_result: TurnResult, turn_con: int):
        if turn_result.passive_player_action_result is None:
            turn_result.passive_player_action_result = ""
        if turn_result.active_player_action_result is None:
            turn_result.active_player_action_result = ""
        if self.dealer.turn_int == turn_con:
            turn_result.active_player_action_result += ""
        else:
            turn_result.passive_player_action_result += ""
            turn_result.give_turn = True

    def use_item(
        self,
        item: Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"],
    ):
        player, target = self.solve_players()

        if player.get_number_of_item(item) == 0:
            return (f"You don't have {item}", None)

        match item:
            case ItemType.HANDSAW.value:
                if not self.dealer.use_items(ItemType.HANDSAW.value):
                    return "❌Only 1 item per turn", None
                player.inventory.handsaw -= 1
                self.shotgun.damage = 2
                return (
                    "🧨Careful, the weapon now deals 2 damage",
                    f"{player.data.name} uses🪚",
                )
            case ItemType.BEER.value:
                if not self.dealer.use_items(ItemType.BEER.value):
                    return "❌Only 1 item per turn", None
                player.inventory.beer -= 1
                round = self.shotgun.shaking()
                return (
                    f"⤴️ It's delicious. The {round} flew out of the shotgun",
                    f"{player.data.name} uses 🍺. The ⤴️{round} flew out of the shotgun ",
                )
            case ItemType.SMOKE.value:
                if not self.dealer.use_items(ItemType.SMOKE.value):
                    return "❌Only 1 item per turn", None
                player.inventory.smoke -= 1
                player.smoke()
                return (
                    f"🚬You now have {player.data.hp} hp",
                    f"{player.data.name} uses 🚬",
                )
            case ItemType.HANDCUFF.value:
                if not self.dealer.use_items(ItemType.HANDCUFF.value):
                    return "❌Only 1 item per turn", None
                if target.tied > 0:
                    return ("You can't now", None)
                player.inventory.handcuff -= 1
                target.tied = 3
                self.dealer.extend_turn()
                return (
                    f"🔗{target.data.name} is tied for 1 turn",
                    f"🔗 {player.data.name} has tied you up",
                )
            case ItemType.GLASS.value:
                if not self.dealer.use_items(ItemType.GLASS.value):
                    return "❌Only 1 item per turn", None
                player.inventory.magnifying_glass -= 1
                return (
                    f"🔍 You see the {str(self.shotgun.inspect())} inside",
                    f"{player.data.name} uses 🔍. Very interesting...",
                )

    def distribute_items(self, player: Player, desired_length: int):
        blackbox = [random.choice(self.ITEMS) for _ in range(desired_length)]

        for item in blackbox:
            try:
                player.add_item(item.value)
            except Exception:
                break

        return player.get_items_emoji()

    def get_current_username(self):
        player, target = self.solve_players()
        return player.data.name

    def solve_players(self):
        if self.dealer.turn_int % 2 == 0:
            player = self.first_player
            target = self.second_player
        else:
            target = self.first_player
            player = self.second_player

        return player, target
