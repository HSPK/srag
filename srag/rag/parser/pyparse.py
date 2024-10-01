import os

from pyparse_client import AsyncPyParse

from srag.schema import Document

from ._base import BaseParser


class PyParser(BaseParser):
    def __init__(self, uri: str, description: str | None = None, base_url: str | None = None):
        super().__init__(uri, description)
        self.client = AsyncPyParse(base_url or os.getenv("PYPARSE_BASE_URL"))

    async def parse(self) -> dict:
        ret = await self.client.parse(self.uri)
        doc = Document.model_validate(ret.model_dump())
        doc.source = self.uri
        doc.description = self.description
        return doc
