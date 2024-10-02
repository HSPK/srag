import os
import time
import warnings
from typing import Literal

from modelhub import AsyncModelhub
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from srag.schema import Chunk, Document, exceptions

from ..retriever._base import BaseRetriever
from ._base import BaseIndexer

warnings.filterwarnings("ignore", module="qdrant_client")


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        indexer: "QdrantIndexer",
        use_full_text: bool = True,
        fusion_method: Literal["rrf", "dbsf"] = "rrf",
        prefetch_top_k: int = 20,
        return_top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.indexer = indexer
        self.qdrant = indexer.qdrant
        self.use_full_text = use_full_text
        self.use_sparse = indexer.use_sparse
        self.prefetch_top_k = prefetch_top_k
        self.return_top_k = return_top_k
        self.score_threshold = score_threshold
        if fusion_method == "rrf":
            self.fusion_method = models.Fusion.RRF
        elif fusion_method == "dbsf":
            self.fusion_method = models.Fusion.DBSF
        else:
            raise ValueError(f"Invalid fusion method: {fusion_method}")

    async def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        index_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        index_name = index_name or self.indexer.default_index_name
        embed = await self.indexer._get_embeddings([query])

        query_kwargs = {
            "score_threshold": self.score_threshold,
            "limit": self.return_top_k,
        }
        doc_id_condition = None
        if doc_ids is not None:
            doc_id_condition = models.FieldCondition(
                key="doc_id", match=models.MatchAny(any=doc_ids)
            )
        if not self.use_full_text and not self.use_sparse:
            query_kwargs["query"] = embed[self.indexer.dense_vector_name][0]
            query_kwargs["using"] = self.indexer.dense_vector_name
        else:
            query_kwargs["query"] = models.FusionQuery(fusion=self.fusion_method)
            query_kwargs["prefetch"] = [
                models.Prefetch(
                    using=self.indexer.dense_vector_name,
                    query=embed[self.indexer.dense_vector_name][0],
                    limit=self.prefetch_top_k,
                    filter=models.Filter(must=doc_id_condition) if doc_id_condition else None,
                )
            ]
            if self.use_sparse:
                sparse_vec = embed[self.indexer.sparse_vector_name][0]
                query_kwargs["prefetch"].append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(sparse_vec.keys()),
                            values=list(sparse_vec.values()),
                        ),
                        using=self.indexer.sparse_vector_name,
                        limit=self.prefetch_top_k,
                        filter=models.Filter(must=doc_id_condition) if doc_id_condition else None,
                    )
                )
            if self.use_full_text:
                conditions = [
                    models.FieldCondition(key="content", match=models.MatchText(text=query))
                ]
                if doc_id_condition:
                    conditions.append(doc_id_condition)
                query_kwargs["prefetch"].append(
                    models.Prefetch(
                        filter=models.Filter(must=conditions),
                        limit=self.prefetch_top_k,
                    )
                )

        query_resp = await self.qdrant.query_points(
            collection_name=index_name,
            query_filter=models.Filter(must=doc_id_condition) if doc_id_condition else None,
            **query_kwargs,
        )
        chunks = []
        for point in query_resp.points:
            chunk = Document.model_validate(point.payload)
            chunk.score = point.score
            chunks.append(chunk)
        return chunks


class QdrantIndexer(BaseIndexer):
    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        text_embedding_model: str = "bge-m3",
        sparse_embedding_model: str = "bge-m3",
        use_sparse: bool = False,
        use_full_text: bool = True,
        batch_size: int = 64,
        parallel: int = 1,
        distance_type: Literal["cosine", "euclidean", "dot", "manhattan"] = "cosine",
        doc_metadata_collection: str = "doc_metadata",
        default_index_name: str = "documents",
        client: AsyncModelhub | None = None,
        float_type: Literal["float16", "float32"] = "float32",
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
    ):
        self.qdrant = AsyncQdrantClient(
            url=url or os.getenv("QDRANT_URL"),
            api_key=api_key or os.getenv("QDRANT_API_KEY"),
        )
        self.text_embedding_model = text_embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.use_sparse = use_sparse
        self.use_full_text = use_full_text
        self.batch_size = batch_size
        self.parallel = parallel
        self.client = client or AsyncModelhub()
        self.doc_metadata_collection = doc_metadata_collection
        self.default_index_name = default_index_name
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        if float_type == "float16":
            self.float_type = models.Datatype.FLOAT16
        elif float_type == "float32":
            self.float_type = models.Datatype.FLOAT32
        else:
            raise ValueError(f"Invalid float type: {float_type}")
        if distance_type == "cosine":
            self.distance_type = models.Distance.COSINE
        elif distance_type == "euclidean":
            self.distance_type = models.Distance.EUCLID
        elif distance_type == "dot":
            self.distance_type = models.Distance.DOT
        elif distance_type == "manhattan":
            self.distance_type = models.Distance.MANHATTAN
        else:
            raise ValueError(f"Invalid distance type: {distance_type}")

    async def _setup_collections(self, doc: Document):
        if not await self.qdrant.collection_exists(self.doc_metadata_collection):
            await self.qdrant.create_collection(
                collection_name=self.doc_metadata_collection,
                vectors_config=models.VectorParams(
                    size=1, datatype=self.float_type, distance=self.distance_type
                ),
            )
            for field, schema in doc.get_index_fieldname_schemas().items():
                await self.qdrant.create_payload_index(
                    collection_name=self.doc_metadata_collection,
                    field_name=field,
                    field_schema=schema,
                )
        self.qdrant.upload_points(
            collection_name=self.doc_metadata_collection,
            points=[
                models.PointStruct(
                    id=doc.id,
                    vector=[0],
                    payload=doc.model_dump(exclude=["chunks"]),
                )
            ],
        )

    async def _get_embeddings(self, texts: list[str]):
        try:
            embed_output = await self.client.get_embeddings(texts, model=self.text_embedding_model)
            embeddings = embed_output.embeddings
            assert len(embeddings) == len(texts)
            if self.use_sparse:
                sparse_embed_output = await self.client.get_embeddings(
                    texts,
                    model=self.sparse_embedding_model,
                    parameters={"return_sparse": True},
                )
                sparse_embeddings = sparse_embed_output.weights
                assert len(sparse_embeddings) == len(texts)
        except Exception as e:
            raise exceptions.IndexEmbeddingException(f"Failed to embed document: {e}")
        ret = {"dense": embeddings, "dense_size": len(embeddings[0])}
        if self.use_sparse:
            ret["sparse"] = sparse_embeddings
        return ret

    async def _embed_chunks(self, chunks: list[Chunk], embed: dict):
        points = []
        for i, chunk in enumerate(chunks):
            vector = {self.dense_vector_name: embed[self.dense_vector_name][i]}
            if self.use_sparse:
                vector[self.sparse_vector_name] = models.SparseVector(
                    indices=list(embed[self.sparse_vector_name][i].keys()),
                    values=list(embed[self.sparse_vector_name][i].values()),
                )
            point = models.PointStruct(
                id=chunk.id,
                vector=vector,
                payload=chunk.model_dump(exclude=["children"]),
            )
            points.append(point)
        return points

    async def _check_index(self, doc: Document, index_name: str, embed_dim: int):
        if await self.qdrant.collection_exists(index_name):
            collection = await self.qdrant.get_collection(index_name)
            act_dim = collection.config.params.vectors[self.dense_vector_name].size
            if act_dim != embed_dim:
                raise exceptions.IndexEmbeddingException(
                    f"Embedding dimension mismatch for collection {index_name}. "
                    f"Expected: {embed_dim}, Actual: {act_dim}"
                )
        else:
            kwargs = {
                "collection_name": index_name,
                "vectors_config": {
                    self.dense_vector_name: models.VectorParams(
                        size=embed_dim,
                        distance=self.distance_type,
                        datatype=self.float_type,
                    )
                },
            }
            if self.use_sparse:
                kwargs["sparse_vectors_config"] = {
                    self.sparse_vector_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(datatype=self.float_type)
                    )
                }
            await self.qdrant.create_collection(**kwargs)
            for field, schema in doc.get_index_fieldname_schemas().items():
                await self.qdrant.create_payload_index(
                    collection_name=index_name,
                    field_name=field,
                    field_schema=schema,
                )

    async def index(self, doc: Document, index_name: str | None = None):
        index_name = index_name or self.default_index_name
        doc.index_name = index_name
        doc.index_time = time.time()
        """Embed and create index for document and it's chunks.

        Args:
            doc (Document): Document to index
        """
        chunks = doc.get_chunks()
        if not chunks:
            return

        chunks_content = [chunk.content for chunk in chunks]
        embed = await self._get_embeddings(chunks_content)
        points = await self._embed_chunks(chunks, embed)

        try:
            await self._check_index(doc, index_name, embed["dense_size"])
            await self._setup_collections(doc)
            self.qdrant.upload_points(
                collection_name=index_name,
                points=points,
                batch_size=self.batch_size,
                parallel=self.parallel,
            )
        except Exception as e:
            raise exceptions.IndexUploadException(
                f"Failed to upload points for document {doc.id}: {e}"
            )

    async def list_index(self):
        """list all indexes"""
        return await self.qdrant.get_collections()

    async def get(
        self,
        doc_id: str,
        chunk_limit: int = 10,
        offset: int | None = None,
        return_chunks: bool = True,
        **kwargs,
    ):
        """Get a document by ID"""
        query_resp = await self.qdrant.scroll(
            collection_name=self.doc_metadata_collection,
            scroll_filter=models.Filter(must=models.HasIdCondition(has_id=[doc_id])),
        )
        if not query_resp or not query_resp[0]:
            return None

        doc = Document.model_validate(query_resp[0][0].payload)
        if not return_chunks:
            return doc
        chunks = await self.qdrant.scroll(
            collection_name=doc.index_name,
            scroll_filter=models.Filter(
                must=models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            ),
            limit=chunk_limit,
            offset=offset,
        )
        doc.chunks = [Chunk.model_validate(chunk.payload) for chunk in chunks[0]]
        return doc

    async def _get_docs(self):
        count = await self.qdrant.count(collection_name=self.doc_metadata_collection)
        scroll_resp = await self.qdrant.scroll(
            collection_name=self.doc_metadata_collection, limit=count.count
        )
        return [Document.model_validate(point.payload) for point in scroll_resp[0]]

    async def get_retriever(self, **kwargs):
        return QdrantRetriever(indexer=self, **kwargs)

    async def delete(self, doc_id: str):
        doc = await self.get(doc_id, return_chunks=False)
        await self.qdrant.delete(
            collection_name=self.doc_metadata_collection,
            points_selector=models.PointIdsList(points=[doc_id]),
        )
        await self.qdrant.delete(
            collection_name=doc.index_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                )
            ),
        )

    async def delete_chunk(self, doc_id: str, chunk_id: str):
        doc = await self.get(doc_id, return_chunks=False)
        await self.qdrant.delete_points(
            collection_name=doc.index_name,
            points_selector=models.PointIdsList(points=[chunk_id]),
        )

    async def update(self, doc_id: str, doc: Document):
        pass

    async def update_chunks(self, doc_id: str, chunks: list[Chunk]):
        doc = await self.get(doc_id, return_chunks=False)
        embed = await self._get_embeddings([chunk.content for chunk in chunks])
        points = await self._embed_chunks(chunks, embed)
        await self.qdrant.batch_update_points(
            collection_name=doc.index_name,
            update_operations=models.UpsertOperation(upsert=models.PointsList(points=points)),
        )

    async def drop(self, index_name: str, **kwargs):
        return await self.qdrant.delete_collection(collection_name=index_name)
