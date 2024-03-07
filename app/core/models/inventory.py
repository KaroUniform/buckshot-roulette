from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
    validator,
)


class InventoryModel(BaseModel):

    model_config = ConfigDict(validate_assignment=True)

    handsaw: int = 0
    beer: int = 0
    smoke: int = 0
    handcuff: int = 0
    magnifying_glass: int = 0

    @model_validator(mode="after")
    def check_total_items(self):
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
