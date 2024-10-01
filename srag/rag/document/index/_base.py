from srag.schema import Chunk, Document


class BaseIndexer:
    """Base class for indexing documents"""

    async def index(self, doc: Document, index_name: str | None = None):
        """Create index for document"""
        pass

    async def get_retriever(self, **kwargs):
        """Get retriever for indexed documents"""
        pass

    async def list_index(self):
        """List all indexed documents"""
        pass

    async def get(self, doc_id: str, **kwargs):
        """Get a document by ID"""
        pass

    async def delete(self, doc_id: str):
        pass

    async def delete_chunk(self, doc_id: str, chunk_id: str):
        pass

    async def update(self, doc_id: str, doc: Document):
        pass

    async def update_chunks(self, doc_id: str, chunks: list[Chunk]):
        pass

    async def drop(self, index_name: str, **kwargs):
        pass
