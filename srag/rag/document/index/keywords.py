from srag.schema import Document

from ._base import BaseIndexer


class KeyWordsIndexer(BaseIndexer):
    async def index(self, docs: list[Document]):
        pass

    async def list_index(self):
        pass
