from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    model_validator,
)


class InventoryModel(BaseModel):
    """
    Represents an inventory model with different types of items.
    """

    model_config = ConfigDict(validate_assignment=True)

    handsaw: int = 0
    beer: int = 0
    smoke: int = 0
    handcuff: int = 0
    magnifying_glass: int = 0
    phone: int = 0
    pills: int = 0
    adrenaline: int = 0
    inverter: int = 0

    def count_items(self):
        return (
            self.handcuff
            + self.beer
            + self.smoke
            + self.handsaw
            + self.magnifying_glass
            + self.phone
            + self.pills
            + self.adrenaline
            + self.inverter
        )


