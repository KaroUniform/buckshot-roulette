from .models.items import ItemType
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
    - add_item(item: ItemType): Adds an item to the player's inventory.
    - delete_item(item: ItemType): Removes an item from the player's inventory.
    - get_number_of_item(item: ItemType) -> int: Retrieves the quantity of a specific item in the player's inventory.
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

    def smoke(self, hp=1):
        """
        Increases the player's health points (HP) by hp if it is less than the maximum HP.
        """
        self.data.hp = min(self.data.hp + hp, self.data.max_hp)
    def set_hp(self, hp: int):
        """
        Sets both the current HP and maximum HP of the player.

        Args:
        - hp (int): The new HP value.
        """
        self.data.hp = hp
        self.data.max_hp = hp

    def add_item(
        self, item: ItemType
    ):
        """
        Adds an item to the player's inventory.

        Args:
        - item (ItemType): The item to be added.
        """
        if self.inventory.count_items() >= 8:
            return
        
        item_attr = item.value
        
        if hasattr(self.inventory, item_attr):
            current_count = getattr(self.inventory, item_attr)
            new_count = current_count + 1
            setattr(self.inventory, item_attr, new_count)

    def delete_item(
        self, item: ItemType
    ):
        """
        Removes an item from the player's inventory.

        Args:
        - item (ItemType): The item to be removed.
        """
        item_attr = item.value

        # Check if the item exists in the inventory and reduce its count.
        if hasattr(self.inventory, item_attr):
            current_count = getattr(self.inventory, item_attr)
            # Ensure the item count cannot go below zero.
            new_count = max(0, current_count - 1)
            setattr(self.inventory, item_attr, new_count)

    def get_number_of_item(
        self, item: ItemType
    ) -> int:
        """
        Retrieves the quantity of a specific item in the player's inventory.

        Args:
        - item (ItemType): The item to check.

        Returns:
        - int: The quantity of the specified item in the player's inventory.
        """
        return self.inventory.__getattribute__(item.value)

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
            "adrenaline": "💉",
            "inverter": "🔀",
            "phone": "📞",
            "pills": "💊"
            
        }

        result = []
        for item, count in blackbox.items():
            if count > 0:
                emoji = emoji_mapping.get(item, None)
                if not emoji:
                    raise ValueError(f"An unexpected item was given: {item}")
                result.append(f"{emoji}x{count}")

        return result

    def take_damage(self, damage: int):
        """
        Reduces the player's HP by the specified amount of damage.

        Args:
        - damage (int): The amount of damage to be applied.
        """
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
