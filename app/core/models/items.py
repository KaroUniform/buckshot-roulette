from pydantic import BaseModel
from enum import Enum


# class ItemType(BaseModel):
#     HANDSAW = "handsaw"
#     BEER = "beer"
#     SMOKE = "smoke"
#     HANDCUFF = "handcuff"
#     GLASS = "magnifying_glass"


class ItemType(Enum):
    HANDSAW = "handsaw"
    BEER = "beer"
    SMOKE = "smoke"
    HANDCUFF = "handcuff"
    GLASS = "magnifying_glass"
