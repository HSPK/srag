from typing import Awaitable, Callable

from srag.rag.document import BaseReranker, BaseRetriever
from srag.schema import BaseTransform, Chunk, LLMCost, Message, RAGState


class TextProcessor(BaseTransform):
    def __init__(self, fn_process: Callable[[str], Awaitable[str]], key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn_process = fn_process
        self.key = key

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state[self.key] = await self.fn_process(state[self.key])
        return state


class HistoryProcessor(BaseTransform):
    def __init__(self, fn_process: Callable[[list[Message]], Awaitable[str]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn_process = fn_process

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state.history = await self.fn_process(state.history)
        return state


class ContextComposer(BaseTransform):
    def __init__(self, fn_process: Callable[[list[Chunk]], Awaitable[str]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn_process = fn_process

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state.context = await self.fn_process(state.chunks)
        return state


class FinalPromptComposer(BaseTransform):
    def __init__(self, fn_process: Callable[[str, str, str], Awaitable[str]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn_process = fn_process

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state.final_prompt = await self.fn_process(
            question=state.query,
            context=state.context,
            history=state.history,
        )
        return state


class FinalGeneration(BaseTransform):
    def __init__(
        self, llm_model: str, temperature: float = 0.01, top_p: float = 0.01, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_p = top_p

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        resp = await self.shared.llm.chat(
            state.final_prompt, model=self.llm_model, temperature=self.temperature, top_p=self.top_p
        )
        state.response = resp.generated_text
        if state.cost is None:
            state.cost = LLMCost()
        i_tokens = resp.details.prompt_tokens or 0
        o_tokens = resp.details.generated_tokens or 0
        state.cost.input_tokens += i_tokens
        state.cost.output_tokens += o_tokens
        state.cost.total_tokens += i_tokens + o_tokens
        return state

    async def stream(self, state: RAGState, **kwargs):
        state.response = ""
        async for token in self.shared.llm.stream_chat(
            state.final_prompt, model=self.llm_model, temperature=self.temperature, top_p=self.top_p
        ):
            state.response += token.token.text
            if token.details.prompt_tokens or token.details.generated_tokens:
                i_tokens = token.details.prompt_tokens or 0
                o_tokens = token.details.generated_tokens or 0
                state.cost.input_tokens += i_tokens
                state.cost.output_tokens += o_tokens
                state.cost.total_tokens += i_tokens + o_tokens
            yield state


class Retriever(BaseTransform):
    def __init__(self, retriever: BaseRetriever, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = retriever

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state.chunks = await self.retriever.retrieve(state.query, state.doc_ids)
        return state


class Reranker(BaseTransform):
    def __init__(self, reranker: BaseReranker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = reranker

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state.chunks = await self.reranker.rerank(state.query, state.chunks)
        return state
