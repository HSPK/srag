import os

from elasticsearch import AsyncElasticsearch

from srag.schema import Document

from ._base import BaseIndexer


class ElasticSearchIndexer(BaseIndexer):
    def __init__(
        self,
        host: str | None = None,
        api_key: str | None = None,
    ):
        self.es = AsyncElasticsearch(
            hosts=host or os.getenv("ELASTIC_SEARCH_HOST"),
            api_key=api_key or os.getenv("ELASTIC_SEARCH_API_KEY"),
        )

    async def index(self, doc: Document, index_name: str | None = None):
        return await super().index(doc, index_name)

    async def list_index(self):
        """list all indexes"""
        return await self.es.indices.get_alias("*")
