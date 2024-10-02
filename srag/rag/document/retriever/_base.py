from loguru import logger
from modelhub import AsyncModelhub

from srag.schema import Chunk


class BaseRetriever:
    async def retrieve(self, query: str, doc_ids: list[str], **kwargs) -> list[Chunk]:
        return []


class BaseReranker:
    async def rerank(self, query: str, chunks: list[Chunk], **kwargs) -> list[Chunk]:
        return chunks


class ModelhubReranker(BaseReranker):
    def __init__(
        self,
        client: AsyncModelhub,
        rerank_model: str,
        rerank_top_k: int = 5,
        score_threshold: int = 0.1,
    ):
        self.client = client
        self.rerank_model = rerank_model
        self.rerank_top_k = rerank_top_k
        self.score_threshold = score_threshold

    async def rerank(self, query: str, chunks: list[Chunk], **kwargs) -> list[Chunk]:
        if not chunks:
            return []
        rerank_output = await self.client.cross_embedding(
            [[query, chunk.content] for chunk in chunks], model=self.rerank_model
        )
        scores = rerank_output.scores
        sorted_docs = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[
            0 : self.rerank_top_k
        ]
        for doc, score in sorted_docs:
            doc.score = score
        filterd_docs = [doc for doc, score in sorted_docs if score > self.score_threshold]
        logger.debug(
            f"rerank finished, #reranked: {len(filterd_docs)}, scores: {[doc.score for doc in filterd_docs]}"
        )
        return filterd_docs
