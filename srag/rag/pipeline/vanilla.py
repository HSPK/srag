from typing import Awaitable, Callable

from modelhub import AsyncModelhub

from srag.rag.document import BaseReranker, BaseRetriever
from srag.schema import BasePipeline, Chunk, Message, TransformListener

from ._consts import DEFAULT_FINAL_PROMPT
from .transforms import (
    ContextComposer,
    Generation,
    HistoryProcessor,
    PromptComposer,
    Reranker,
    Retriever,
    TextProcessor,
)


async def default_fn_preprocess(text: str) -> str:
    return text.strip()


async def default_fn_postprocess(text: str) -> str:
    return text.strip()


async def default_fn_history(messages: list[Message]) -> str:
    if not messages:
        return ""
    msg = "\n".join([f"{m.role}: {m.content.strip()}" for m in messages]).strip()
    msg = f"<history>\n{msg}\n</history>"
    return msg


async def default_fn_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return ""
    context = "\n".join([chunk.content for chunk in chunks]).strip()
    context = f"<context>\n{context}\n</context>"
    return context


async def default_fn_final_prompt(query: str, context: str, history: str) -> str:
    return DEFAULT_FINAL_PROMPT.format(question=query, context=context, history=history)


def _build_vanilla_transforms(
    llm_model: str,
    retriever: BaseRetriever | None = None,
    reranker: BaseReranker | None = None,
    fn_preprocess: Callable[[str], Awaitable[str]] | None = None,
    fn_postprocess: Callable[[str], Awaitable[str]] | None = None,
    fn_history: Callable[[list[Message]], Awaitable[str]] | None = None,
    fn_context: Callable[[list[Chunk]], Awaitable[str]] | None = None,
    fn_final_prompt: Callable[[str, str, str], Awaitable[str]] | None = None,
    temperature: float = 0.01,
    top_p: float = 0.01,
):
    transforms = [
        TextProcessor(fn_preprocess or default_fn_preprocess, key="query"),
        HistoryProcessor(fn_history or default_fn_history),
        Retriever(retriever or BaseRetriever()),
        Reranker(reranker or BaseReranker()),
        ContextComposer(fn_process=fn_context or default_fn_context),
        PromptComposer(fn_process=fn_final_prompt or default_fn_final_prompt),
        Generation(llm_model=llm_model, temperature=temperature, top_p=top_p),
        TextProcessor(fn_postprocess or default_fn_postprocess, key="response"),
    ]
    return transforms


def build_vanilla_pipeline(
    llm_model: str,
    llm: AsyncModelhub | None = None,
    retriever: BaseRetriever | None = None,
    reranker: BaseReranker | None = None,
    fn_preprocess: Callable[[str], Awaitable[str]] | None = None,
    fn_postprocess: Callable[[str], Awaitable[str]] | None = None,
    fn_history: Callable[[list[Message]], Awaitable[str]] | None = None,
    fn_context: Callable[[list[Chunk]], Awaitable[str]] | None = None,
    fn_final_prompt: Callable[[str, str, str], Awaitable[str]] | None = None,
    listeners: list[TransformListener] = [],
    temperature: float = 0.01,
    top_p: float = 0.01,
):
    transforms = _build_vanilla_transforms(
        llm_model=llm_model,
        retriever=retriever,
        reranker=reranker,
        fn_preprocess=fn_preprocess,
        fn_postprocess=fn_postprocess,
        fn_history=fn_history,
        fn_context=fn_context,
        fn_final_prompt=fn_final_prompt,
        temperature=temperature,
        top_p=top_p,
    )
    return BasePipeline(transforms, listeners=listeners, llm=llm)
