from srag.schema import Document


class BaseChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunking(self, doc: Document) -> Document:
        raise NotImplementedError
