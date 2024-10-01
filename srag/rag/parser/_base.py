import os

from srag.schema import Document


class BaseParser:
    def __init__(self, uri: str, description: str | None = None, **kwargs) -> None:
        self.uri = uri
        self.description = description or f"File: {os.path.basename(uri)}"

    async def parse(self) -> Document:
        raise NotImplementedError
