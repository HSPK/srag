from typing import Literal

import pydantic

from srag.utils import get_current_time_formatted

from .._base import BaseModel


class Message(BaseModel):
    id: str | None = None
    role: Literal["user", "assistant", "tool"] | str
    content: str
    created_at: str = pydantic.Field(default_factory=get_current_time_formatted)
    metadata: dict | None = None
