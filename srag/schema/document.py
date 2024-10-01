import uuid
from typing import Literal, Union

import pydantic

from srag.utils import get_current_time_formatted

from ._base import BaseModel


class Document(BaseModel):
    """document 由多个 chunk 组成"""

    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    """UUID"""
    source: str | None = None
    description: str | None = None
    num_chunks: int | None = None
    child_chunk_ids: list[str] | None = None
    created_at: str = pydantic.Field(default_factory=get_current_time_formatted)
    updated_at: str = pydantic.Field(default_factory=get_current_time_formatted)
    index_name: str | None = None
    index_time: float | None = None
    metadata: dict | None = None

    chunks: list["Chunk"] = []

    def get_index_fieldname_schemas(self):
        return {
            "id": "uuid",
            "description": "text",
            "created_at": "datetime",
            "updated_at": "datetime",
        }

    def get_chunks(self) -> list["Chunk"]:
        flatten_chunks = []
        for chunk in self.chunks:
            if chunk.children:
                flatten_chunks.extend(chunk.children)
            else:
                flatten_chunks.append(chunk)
        return flatten_chunks

    def add_chunk(self, chunk: Union["Chunk", list["Chunk"]]):
        if isinstance(chunk, list):
            self.chunks.extend(chunk)
        else:
            self.chunks.append(chunk)
        self.num_chunks = len(self.chunks)
        self.child_chunk_ids = [c.id for c in self.chunks]

    def clear_chunks(self):
        self.chunks = []
        self.num_chunks = 0
        self.child_chunk_ids = []


class Chunk(BaseModel):
    """chunk 由多个 token 组成"""

    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    """chunk UUID"""
    index: int | None = None
    """index of the chunk in the document"""
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] | None = None
    doc_id: str | None = None
    """document ID"""
    chunk_type: (
        Literal[
            "text",
            "image",
            "table_desc",
            "container",
            "markdown",
            "markdown_code",
            "markdown_title",
            "markdown_section_header",
            "table_csv",
        ]
        | str
    ) = "text"
    """chunk type"""
    content: str | None = None
    """chunk content"""
    image_name: str | None = None
    image_content: str | None = None
    """encoded image content"""
    table_content: str | None = None
    """table content, decided by chunk_type"""
    num_tokens: int | None = None
    """number of tokens"""
    score: float | None = None
    """chunk relevance score"""
    source: str | None = None
    """chunk source"""
    description: str | None = None
    """chunk description"""
    created_at: str = pydantic.Field(default_factory=get_current_time_formatted)
    """chunk creation time"""
    updated_at: str = pydantic.Field(default_factory=get_current_time_formatted)
    """chunk update time"""
    metadata: dict | None = None
    """Othre metadata"""

    children: list["Chunk"] = []

    def get_index_fieldname_schemas(self):
        return {
            "id": "uuid",
            "parent_chunk_id": "uuid",
            "doc_id": "uuid",
            "chunk_type": "keyword",
            "content": {"type": "text", "tokenizer": "multilingual"},
            "created_at": "datetime",
            "updated_at": "datetime",
        }

    def get_content_to_embed(self):
        return self.content
