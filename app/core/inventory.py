from .models.inventory import InventoryModel


class Inventory:
    def __init__(self, model: InventoryModel) -> None:
        self.data = model
