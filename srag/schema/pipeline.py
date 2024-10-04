from dataclasses import dataclass
from typing import List, Optional, TypedDict, Union, Unpack

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

    async def on_enter(self, *args, **kwargs):
        pass

    async def on_exit(self, *args, **kwargs):
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
        self._inited = False

    async def _init_sub_transforms(self):
        _to_init = [v for _, v in self.__dict__.items() if isinstance(v, BaseTransform)]
        _to_init = _to_init + self._transforms if self._transforms is not None else []
        if self._transforms is not None:
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
                async for s in t.__stream__(state):
                    yield s

    async def __call__(self, state: RAGState, **kwargs):
        await self._init()
        await self.shared.listener.on_transform_enter(self, state)
        state = await self._run_sub_transforms(state)
        ret = await self.transform(state, **kwargs)
        await self.shared.listener.on_transform_exit(self, state)
        return ret

    async def __stream__(self, state: RAGState, **kwargs):
        await self._init()
        await self.shared.listener.on_transform_enter(self, state)
        async for s in self._run_sub_streams(state):
            yield s
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

    async def transform(self, state: RAGState, return_state: bool = False, **kwargs) -> RAGState:
        return state if return_state else state.get(self.output_key)

    async def stream(self, **kwargs):
        async for state in super().__stream__(state=kwargs):
            yield state

    async def stream_transform(self, state: RAGState, **kwargs):
        yield state
