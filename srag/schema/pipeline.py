from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict, Union, Unpack

import anyio
from modelhub import AsyncModelhub

from .document import Chunk
from .llm.message import Message


@dataclass
class LLMCost:
    total_tokens: int = 0
    total_cost: float = 0.0
    input_tokens: int = 0
    input_cost: float = 0.0
    output_tokens: int = 0
    output_cost: float = 0.0


class RAGState(TypedDict, total=False):
    query: str
    doc_ids: List[str]
    rewritten_queries: List[str]
    history: Union[List[Message], str]
    chunks: List[Chunk]
    context: str
    final_prompt: str
    response: str
    cost: LLMCost


@dataclass
class SharedResource:
    llm: AsyncModelhub
    listener: "TranformBatchListener"


class TransformListener:
    async def on_transform_enter(self, transform: "BaseTransform", state: RAGState):
        pass

    async def on_transform_exit(self, transform: "BaseTransform", state: RAGState):
        pass


class TranformBatchListener:
    def __init__(self, listeners: list[TransformListener]):
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


class BaseTransform:
    def __init__(
        self,
        transforms: Optional[List["BaseTransform"]] = None,
        run_in_parallel: bool = False,
        run_type: Literal["before", "after", "ignore"] = "ignore",
        input_key: Optional[Union[List[str], str]] = None,
        output_key: Optional[Union[List[str], str]] = None,
        shared: Optional[SharedResource] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__
        self.input_key = input_key
        self.output_key = output_key
        self.shared = shared

        self._transforms = transforms
        self._run_in_parallel = run_in_parallel
        self._run_type = run_type
        if self._run_type == "ignore" and self._transforms is None:
            self._run_type = "after"
        self._inited = False

    async def _init_sub_transforms(self):
        _to_init = [v for v in self.__dict__.values() if isinstance(v, BaseTransform)]
        if self._transforms is not None:
            _to_init = _to_init + self._transforms
        async with anyio.create_task_group() as tg:
            for t in _to_init:
                t.name = f"{self.name}::{t.name}"
                tg.start_soon(t._init, self.shared)

    async def _init(self, shared: SharedResource | None = None):
        if self._inited:
            return
        if self.shared is None and shared is None:
            raise RuntimeError("SharedResource not provided.")
        self.shared = shared or self.shared
        await self._init_sub_transforms()
        self._inited = True

    def _get_input(self, state: RAGState):
        if isinstance(self.input_key, list):
            return {k: state.get(k) for k in self.input_key}
        else:
            return {self.input_key: state.get(self.input_key)}

    async def _run_sub_transforms(self, state: RAGState):
        if self._transforms is None:
            return state
        if self._run_in_parallel:
            async with anyio.create_task_group() as tg:
                for t in self._transforms:
                    tg.start_soon(t.__call__, state)
        else:
            for t in self._transforms:
                state = await t.__call__(state)
        return state

    async def _run_sub_streams(self, state: RAGState):
        if self._transforms is None:
            return
        if self._run_in_parallel:
            async with anyio.create_task_group() as tg:
                for t in self._transforms:
                    tg.start_soon(t.__call__, state)
            yield state
            return
        else:
            for t in self._transforms:
                async for s in t.stream(state):
                    yield s

    async def __call__(self, state: RAGState, **kwargs):
        await self._init()
        await self.shared.listener.on_transform_enter(self, state)
        if self._run_type == "before":
            state = await self.transform(state, **kwargs)
        state = await self._run_sub_transforms(state)
        if self._run_type == "after":
            state = await self.transform(state, **kwargs)
        await self.shared.listener.on_transform_exit(self, state)
        return state

    async def stream(self, state: RAGState, **kwargs):
        await self._init()
        await self.shared.listener.on_transform_enter(self, state)
        if self._run_type == "before":
            async for s in self.stream_transform(state, **kwargs):
                yield s
        async for s in self._run_sub_streams(state):
            yield s
        if self._run_type == "after":
            async for s in self.stream_transform(state, **kwargs):
                yield s
        await self.shared.listener.on_transform_exit(self, state)
        return

    async def transform(self, state: RAGState, **kwargs) -> RAGState:
        return state

    async def stream_transform(self, state: RAGState, **kwargs):
        yield await self.transform(state)


class BasePipeline(BaseTransform):
    def __init__(
        self,
        transforms: List[BaseTransform] | None = None,
        input_key: List[str] = ["query", "history", "doc_ids"],
        output_key: str = "response",
        listeners: list[TransformListener] | None = None,
        llm: AsyncModelhub | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            transforms=transforms,
            run_in_parallel=False,
            run_type="after",
            input_key=input_key,
            output_key=output_key,
            shared=SharedResource(
                llm=llm or AsyncModelhub(), listener=TranformBatchListener(listeners)
            ),
            *args,
            **kwargs,
        )
        self.forward = self.__call__

    async def __call__(self, return_state: bool = False, **kwargs: Unpack[RAGState]):
        return await super().__call__(state=kwargs, return_state=return_state)

    async def stream(self, **kwargs: Unpack[RAGState]):
        async for state in super().stream(state=kwargs):
            yield state

    async def transform(self, state: RAGState, return_state: bool = False, **kwargs) -> RAGState:
        return state if return_state else state.get(self.output_key)

    async def stream_transform(self, state: RAGState, **kwargs):
        yield state
