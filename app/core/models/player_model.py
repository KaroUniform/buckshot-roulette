from typing import Annotated
from pydantic import BaseModel, ConfigDict, StringConstraints, Field


class PlayerModel(BaseModel):

    model_config = ConfigDict(validate_assignment=True)

    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=25, min_length=1),
    ]
    chat_id: int
    hp: Annotated[int, Field(ge=0, le=6)]
    max_hp: Annotated[int, Field(ge=0, le=6)]
