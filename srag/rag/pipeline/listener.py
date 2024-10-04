import time

import pydantic

from srag.schema import BaseTransform, RAGState, TransformListener


class TransformLog(pydantic.BaseModel):
    name: str
    created_at: float = pydantic.Field(default_factory=lambda: time.time())
    duration: float = 0.0
    state_before: RAGState | None = None
    state_after: RAGState | None = None

    def finish(self, state: RAGState):
        self.duration = time.time() - self.created_at
        self.state_after = state


class PerfTracker(TransformListener):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline_time = 0
        self.time = []

    async def on_transform_enter(self, transform: BaseTransform, state: RAGState):
        self.time.append(time.time())

    async def on_transform_exit(self, transform: BaseTransform, state: RAGState):
        last_time = self.time.pop()
        print(f"{transform.name} used {time.time() - last_time:.2f}s")


class PipelineMemoryStore(TransformListener):
    def __init__(self):
        self.logs = []

    async def on_transform_enter(self, transform: BaseTransform, state: RAGState):
        self.logs.append(TransformLog(name=transform.name, state_before=state))

    async def on_transform_exit(self, transform: BaseTransform, state: RAGState):
        self.logs[-1].finish(state)
