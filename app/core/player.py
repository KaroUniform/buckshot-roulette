from typing import Literal
from .models.player_model import PlayerModel
from .models.inventory import InventoryModel


class Player:

    tied = 0

    def __init__(self, model: PlayerModel) -> None:
        self.data = model
        self.inventory = InventoryModel()

    def flush_inventory(self):
        self.inventory = InventoryModel()

    def smoke(self):
        if self.data.hp < self.data.max_hp:
            self.data.hp += 1

    def set_hp(self, hp: int):
        self.data.hp = hp
        self.data.max_hp = hp

    def add_item(
        self, item: Literal["handsaw", "beer", "smoke", "handcuff", "magnifying_glass"]
    ):
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
    ):
        return self.inventory.__getattribute__(item)

    def get_items_emoji(self):
        return self._emoji_builder(self.inventory.model_dump())

    def _emoji_builder(self, blackbox: dict):
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
        if self.data.hp >= 1:
            self.data.hp -= damage

    def still_alive(self) -> bool:
        return bool(self.data.hp)

    def still_tied(self):
        if self.tied == 2:
            self.tied -= 1
            return True
        else:
            self.tied -= 1
            return False
