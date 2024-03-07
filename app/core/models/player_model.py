from typing import Annotated
from pydantic import BaseModel, ConfigDict, StringConstraints, Field


class PlayerModel(BaseModel):
    """
    Represents a player model with attributes such as name, chat ID, health points (HP), and maximum HP.

    Attributes:
    - model_config (ConfigDict): Configuration settings for the model with validate_assignment=True.
    - name (str): The name of the player, constrained by maximum and minimum lengths and stripped of whitespace.
    - chat_id (int): The unique identifier for the player's chat (Use telegram chat_id for telegram).
    - hp (int): The current health points (HP) of the player, constrained to be between 0 and 6.
    - max_hp (int): The maximum health points (HP) a player can have, constrained to be between 0 and 6.

    Usage:
    Create an instance of PlayerModel by providing values for the attributes.
    The attributes are automatically validated based on the specified constraints.

    Example:
    >>> player_data = PlayerModel(name="JohnDoe", chat_id=123456, hp=3, max_hp=6)
    """

    model_config = ConfigDict(validate_assignment=True)

    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=25, min_length=1),
    ]
    chat_id: int
    hp: Annotated[int, Field(ge=0, le=6)]
    max_hp: Annotated[int, Field(ge=0, le=6)]
