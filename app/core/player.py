from typing import Literal
from .models.player_model import PlayerModel
from .models.inventory import InventoryModel


class Player:
    """
    Represents a player in the game with attributes such as data, inventory, and tied status.

    Attributes:
    - tied (int): Counter representing the tied status of the player.
    - data (PlayerModel): PlayerModel instance containing player information.
    - inventory (InventoryModel): InventoryModel instance representing the player's inventory.

    Methods:
    - flush_inventory(): Resets the player's inventory by creating a new InventoryModel instance.
    - smoke(): Increases the player's health points (HP) by 1 if it is less than the maximum HP.
    - set_hp(hp: int): Sets both the current HP and maximum HP of the player.
    - add_item(item: Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]): Adds an item to the player's inventory.
    - delete_item(item: Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]): Removes an item from the player's inventory.
    - get_number_of_item(item: Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]) -> int: Retrieves the quantity of a specific item in the player's inventory.
    - get_items_emoji() -> list: Retrieves a list of emoji representations for the items in the player's inventory.
    - take_damage(damage: int): Reduces the player's HP by the specified amount of damage.
    - still_alive() -> bool: Checks if the player is still alive based on their current HP.
    - still_tied() -> bool: Decreases the tied status of the player and returns True if still tied, False otherwise.
    """

    tied = 0

    def __init__(self, model: PlayerModel) -> None:
        """
        Initializes a Player with a PlayerModel instance and a new InventoryModel.
        """
        self.data = model
        self.inventory = InventoryModel()

    def flush_inventory(self):
        """
        Resets the player's inventory by creating a new InventoryModel instance.
        """
        self.inventory = InventoryModel()

    def smoke(self):
        """
        Increases the player's health points (HP) by 1 if it is less than the maximum HP.
        """
        if self.data.hp < self.data.max_hp:
            self.data.hp += 1

    def set_hp(self, hp: int):
        """
        Sets both the current HP and maximum HP of the player.

        Args:
        - hp (int): The new HP value.
        """
        self.data.hp = hp
        self.data.max_hp = hp

    def add_item(
        self, item: Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]
    ):
        """
        Adds an item to the player's inventory.

        Args:
        - item (Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]): The item to be added.
        """
        if self.inventory.count_items() >= 8:
            return

        match item:
            case "handsaw":
                self.inventory.handsaw += 1
            case "beer":
                self.inventory.beer += 1
            case "smoke":
                self.inventory.smoke += 1
            case "handcuff":
                self.inventory.handcuff += 1
            case "magnifying_glass":
                self.inventory.magnifying_glass += 1

    def delete_item(
        self, item: Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]
    ):
        """
        Removes an item from the player's inventory.

        Args:
        - item (Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]): The item to be removed.
        """
        match item:
            case "handsaw":
                self.inventory.handsaw -= 1
            case "beer":
                self.inventory.beer -= 1
            case "smoke":
                self.inventory.smoke -= 1
            case "handcuff":
                self.inventory.handcuff -= 1
            case "magnifying_glass":
                self.inventory.magnifying_glass -= 1

    def get_number_of_item(
        self, item: Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]
    ) -> int:
        """
        Retrieves the quantity of a specific item in the player's inventory.

        Args:
        - item (Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]): The item to check.

        Returns:
        - int: The quantity of the specified item in the player's inventory.
        """
        return self.inventory.__getattribute__(item)

    def get_items_emoji(self) -> list:
        """
        Retrieves a list of emoji representations for the items in the player's inventory.

        Returns:
        - list: A list of emoji representations for the items in the player's inventory.
        """
        return self._emoji_builder(self.inventory.model_dump())

    def _emoji_builder(self, blackbox: dict):
        """
        Builds a list of emoji representations for the items in the player's inventory.

        Args:
        - blackbox (dict): The player's inventory as a dictionary.

        Returns:
        - list: A list of emoji representations for the items in the player's inventory.
        """
        emoji_mapping = {
            "handsaw": "🪚",
            "beer": "🍺",
            "smoke": "🚬",
            "handcuff": "🔗",
            "magnifying_glass": "🔍",
        }

        result = []
        for item, count in blackbox.items():
            if count > 0:
                emoji = emoji_mapping.get(item, "")
                if emoji:
                    result.append(f"{emoji}x{count}")

        return result

    def take_damage(self, damage: int):
        """
        Reduces the player's HP by the specified amount of damage.

        Args:
        - damage (int): The amount of damage to be applied.
        """
        if self.data.hp >= 1:
            self.data.hp -= damage

    def still_alive(self) -> bool:
        """
        Checks if the player is still alive based on their current HP.

        Returns:
        - bool: True if the player is still alive, False otherwise.
        """
        return bool(self.data.hp)

    def still_tied(self) -> bool:
        """
        Decreases the tied status of the player and returns True if still tied, False otherwise.

        Returns:
        - bool: True if the player is still tied, False otherwise.
        """
        if self.tied == 2:
            self.tied -= 1
            return True
        else:
            self.tied -= 1
            return False
