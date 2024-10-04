from typing import Awaitable, Callable

from srag.rag.document import BaseReranker, BaseRetriever
from srag.schema import BaseTransform, Chunk, LLMCost, Message, RAGState


class TextProcessor(BaseTransform):
    def __init__(self, fn_process: Callable[[str], Awaitable[str]], key: str):
        super().__init__(input_key=key, output_key=key)
        self.fn_process = fn_process
        self.key = key

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state[self.key] = await self.fn_process(state[self.key])
        return state


class HistoryProcessor(BaseTransform):
    def __init__(
        self,
        fn_process: Callable[[list[Message]], Awaitable[str]],
        input_key: str = "history",
        output_key: str = "history",
    ):
        super().__init__(input_key=input_key, output_key=output_key)
        self.fn_process = fn_process

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state[self.output_key] = await self.fn_process(state.get(self.input_key))
        return state


class ContextComposer(BaseTransform):
    def __init__(
        self,
        fn_process: Callable[[list[Chunk]], Awaitable[str]],
        input_key: str = "chunks",
        output_key: str = "context",
    ):
        super().__init__(input_key=input_key, output_key=output_key)
        self.fn_process = fn_process

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state[self.output_key] = await self.fn_process(state.get(self.input_key))
        return state


class PromptComposer(BaseTransform):
    def __init__(
        self,
        fn_process: Callable[[str, str, str], Awaitable[str]],
        input_key: list[str] = ["query", "context", "history"],
        output_key: str = "final_prompt",
    ):
        super().__init__(input_key=input_key, output_key=output_key)
        self.fn_process = fn_process

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state[self.output_key] = await self.fn_process(**self._get_input(state))
        return state


class Generation(BaseTransform):
    def __init__(
        self,
        llm_model: str,
        temperature: float = 0.01,
        top_p: float = 0.01,
        max_tokens: int | None = None,
        input_key: str = "final_prompt",
        output_key: str = "response",
        cost_key: str = "cost",
    ):
        super().__init__(input_key=input_key, output_key=output_key)
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.input_key = input_key
        self.cost_key = cost_key

    def _prepare_chat_kwargs(self, state: RAGState):
        chat_kwargs = {
            "prompt": state[self.input_key],
            "model": self.llm_model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        return {k: v for k, v in chat_kwargs.items() if v is not None}

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        resp = await self.shared.llm.chat(**self._prepare_chat_kwargs(state))
        state[self.output_key] = resp.generated_text
        cost = state.get(self.cost_key, LLMCost())
        i_tokens = resp.details.prompt_tokens or 0
        o_tokens = resp.details.generated_tokens or 0
        cost.input_tokens += i_tokens
        cost.output_tokens += o_tokens
        cost.total_tokens += i_tokens + o_tokens
        state[self.cost_key] = cost
        return state

    async def stream_transform(self, state: RAGState, **kwargs):
        state[self.output_key] = ""
        async for token in self.shared.llm.stream_chat(**self._prepare_chat_kwargs(state)):
            state[self.output_key] += token.token.text
            if token.details.prompt_tokens or token.details.generated_tokens:
                cost = state.get(self.cost_key, LLMCost())
                i_tokens = token.details.prompt_tokens or 0
                o_tokens = token.details.generated_tokens or 0
                cost.input_tokens += i_tokens
                cost.output_tokens += o_tokens
                cost.total_tokens += i_tokens + o_tokens
                state[self.cost_key] = cost
            yield state


class Retriever(BaseTransform):
    def __init__(
        self,
        retriever: BaseRetriever,
        input_key: list[str] = ["query", "doc_ids"],
        output_key: str = "chunks",
    ):
        super().__init__(input_key=input_key, output_key=output_key)
        self.retriever = retriever

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state[self.output_key] = await self.retriever.retrieve(**self._get_input(state))
        return state


class Reranker(BaseTransform):
    def __init__(
        self,
        reranker: BaseReranker,
        input_key: list[str] = ["query", "chunks"],
        output_key: str = "chunks",
    ):
        super().__init__(input_key=input_key, output_key=output_key)
        self.reranker = reranker

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        state[self.output_key] = await self.reranker.rerank(**self._get_input(state))
        return state
