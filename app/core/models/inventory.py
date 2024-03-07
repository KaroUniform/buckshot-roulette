from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    model_validator,
)


class InventoryModel(BaseModel):
    """
    Represents an inventory model with different types of items.

    Attributes:
    - model_config (ConfigDict): Configuration settings for the model with validate_assignment=True.
    - handsaw (int): Quantity of handsaw items in the inventory.
    - beer (int): Quantity of beer items in the inventory.
    - smoke (int): Quantity of smoke items in the inventory.
    - handcuff (int): Quantity of handcuff items in the inventory.
    - magnifying_glass (int): Quantity of magnifying glass items in the inventory.

    Methods:
    - check_total_items(): Validates that the total number of items in the inventory does not exceed 8.

    Usage:
    Create an instance of InventoryModel and set the quantities of different items.
    Use the check_total_items method to ensure the inventory stays within the specified limits.

    Example:
    >>> inventory = InventoryModel(handsaw=2, beer=3, smoke=1, handcuff=1, magnifying_glass=1)
    """

    model_config = ConfigDict(validate_assignment=True)

    handsaw: int = 0
    beer: int = 0
    smoke: int = 0
    handcuff: int = 0
    magnifying_glass: int = 0

    def count_items(self):
        return (
            self.handcuff
            + self.beer
            + self.smoke
            + self.handsaw
            + self.magnifying_glass
        )

    @model_validator(mode="after")
    def check_total_items(self):
        """
        Validates that the total number of items in the inventory does not exceed 8.

        Returns:
        - InventoryModel: The instance of the inventory model if validation passes.

        Raises:
        - ValidationError: If the total number of items exceeds 8.
        """
        total_items = (
            self.handcuff
            + self.beer
            + self.smoke
            + self.handsaw
            + self.magnifying_glass
        )
        if total_items > 8:
            raise ValidationError(
                "The total number of items in the inventory cannot exceed 8."
            )
        return self
