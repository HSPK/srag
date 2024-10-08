from srag.schema import Chunk, Document

from ._base import BaseChunker


class MarkdownChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 5, minimum_chunk_size: int = 64):
        super().__init__(chunk_size, overlap)
        self.minimum_chunk_size = minimum_chunk_size

    async def estimate_tokens(self, text):
        return int(len(text) * 1.5)

    async def split_chunk(self, chunk: Chunk) -> list[Chunk]:
        num_tokens = await self.estimate_tokens(chunk.content)
        if num_tokens <= self.chunk_size:
            return [chunk]
        if chunk.chunk_type == "table_csv":
            table_header, table_body = chunk.table_content.split("\n", 1)
            chunk_pos = 0
            new_chunks = []
            while chunk_pos < len(table_body):
                next_chunk_pos = min(chunk_pos + self.chunk_size, len(table_body))
                while (
                    table_body[next_chunk_pos - 1] != "\n"
                    and next_chunk_pos > chunk_pos + self.minimum_chunk_size
                ):
                    # find the last line break, but not too far
                    next_chunk_pos -= 1
                new_content = f"{table_header}\n{table_body[chunk_pos:next_chunk_pos]}"
                new_chunk = Chunk(
                    doc_id=chunk.doc_id,
                    chunk_type=chunk.chunk_type,
                    content=new_content,
                    metadata=chunk.metadata,
                )
                chunk_pos = next_chunk_pos
                new_chunks.append(new_chunk)
            return new_chunks
        else:
            new_chunks = []
            for i in range(0, len(chunk.content), self.chunk_size):
                new_chunks.append(
                    Chunk(
                        doc_id=chunk.doc_id,
                        chunk_type=chunk.chunk_type,
                        content=chunk.content[i : i + self.chunk_size],
                        metadata=chunk.metadata,
                    )
                )
            print(f"Split chunk into {len(new_chunks)} chunks")
            return new_chunks

    async def chunking(self, doc: Document) -> Document:
        current_header = None
        current_title = None
        new_chunks: list[Chunk] = []
        for chunk in doc.chunks:
            if chunk.chunk_type == "markdown_title":
                current_title = chunk.content.strip()
                continue
            elif chunk.chunk_type == "markdown_section_header" or chunk.content.strip().startswith(
                "##"
            ):
                current_header = chunk.content.strip()
                continue
            if chunk.chunk_type == "table_csv":
                # use csv content for chunking
                chunk.content = chunk.table_content
            split_chunks = await self.split_chunk(chunk)
            for c in split_chunks:
                if current_header:
                    c.content = f"Section: {current_header}\n{c.content.strip()}"
                if current_title:
                    c.content = f"Title: {current_title}\n{c.content.strip()}"
                c.content = c.content.strip()
                c.num_tokens = await self.estimate_tokens(c.content)
            new_chunks.extend(split_chunks)
        for i in range(len(new_chunks)):
            new_chunks[i].index = i
        new_doc = doc.model_copy()
        new_doc.clear_chunks()
        new_doc.add_chunk(new_chunks)
        return new_doc
