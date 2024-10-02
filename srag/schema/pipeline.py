from dataclasses import dataclass

import anyio
import pydantic
from modelhub import AsyncModelhub

from .document import Chunk
from .llm.message import Message


class LLMCost(pydantic.BaseModel):
    total_tokens: int = 0
    total_cost: float = 0.0
    input_tokens: int = 0
    input_cost: float = 0.0
    output_tokens: int = 0
    output_cost: float = 0.0


class RAGState(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="allow")

    query: str | None = None
    doc_ids: list[str] | None = None
    rewritten_queries: list[str] | None = None
    history: list[Message] | str | None = None
    chunks: list[Chunk] | None = None
    context: str | None = None
    final_prompt: str | None = None
    response: str | None = None
    cost: LLMCost | None = None

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            v = self.model_dump().get(name, None)
            if v is not None:
                return v
            raise AttributeError(f"Attribute {name} not found.") from e

    def __getitem__(self, name: str):
        return self.__getattribute__(name)

    def __setitem__(self, name: str, value):
        self.__setattr__(name, value)


class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


@dataclass
class SharedResource:
    llm: AsyncModelhub | None = None


class BaseTransform:
    _inited: bool = False
    name: str = "transform"
    shared: SharedResource | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    async def _init(self, shared: SharedResource):
        """Internal initialization, not to be called directly."""
        self.shared = shared
        self._inited = True

    async def _transform(self, *args, **kwargs) -> RAGState:
        if not self._inited:
            raise RuntimeError("Transformer not initialized.")
        return await self.transform(*args, **kwargs)

    async def _stream(self, *args, **kwargs):
        if not self._inited:
            raise RuntimeError("Transformer not initialized.")

        async for s in self.stream(*args, **kwargs):
            yield s

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        raise NotImplementedError

    async def stream(self, state: RAGState, **kwargs):
        yield await self.transform(state, **kwargs)


class PipelineListener:
    async def on_transform_enter(self, transform: BaseTransform, state: RAGState):
        pass

    async def on_transform_exit(self, transform: BaseTransform, state: RAGState):
        pass

    async def on_enter(self, *args, **kwargs):
        pass

    async def on_exit(self, *args, **kwargs):
        pass


class PipelineBatchListener:
    def __init__(self, listeners: list[PipelineListener]):
        if listeners is None:
            listeners = []
        self.listeners = listeners

    def _on_event_construct(self, event: str):
        async def _on_event(*args):
            if not self.listeners:
                return
            async with anyio.create_task_group() as tg:
                for listener in self.listeners:
                    tg.start_soon(listener.__getattribute__(event), *args)

        return _on_event

    def __getattribute__(self, name: str):
        if name.startswith("on_"):
            return self._on_event_construct(name)
        return super().__getattribute__(name)


class BasePipeline:
    def __init__(
        self,
        transforms: list[BaseTransform],
        *,
        listeners: list[PipelineListener] | None = None,
        llm: AsyncModelhub | None = None,
    ):
        self.llm = llm or AsyncModelhub()
        self._transforms = transforms
        self._listener = PipelineBatchListener(listeners)
        self._inited = False

    async def _init(self):
        if self._inited:
            return
        async with anyio.create_task_group() as tg:
            """Initialize all the transforms."""
            for t in self._transforms:
                tg.start_soon(t._init, SharedResource(llm=self.llm))
        self._inited = True

    async def stream(self, **kwargs):
        await self._init()
        await self._listener.on_enter()
        state = RAGState(**kwargs)
        for t in self._transforms:
            await self._listener.on_transform_enter(t, state)
            async for s in t._stream(state):
                yield s
            await self._listener.on_transform_exit(t, state)
        await self._listener.on_exit()

    async def __call__(self, return_state: bool = False, **kwargs):
        await self._init()
        await self._listener.on_enter()
        state = RAGState(**kwargs)
        for t in self._transforms:
            await self._listener.on_transform_enter(t, state)
            state = await t._transform(state)
            await self._listener.on_transform_exit(t, state)
        await self._listener.on_exit()
        return state if return_state else state.response
