import time

import pydantic

from srag.schema import BaseTransform, PipelineListener, RAGState


class TransformLog(pydantic.BaseModel):
    name: str
    created_at: float = pydantic.Field(default_factory=lambda: time.time())
    duration: float = 0.0
    state_before: RAGState | None = None
    state_after: RAGState | None = None

    def finish(self, state: RAGState):
        self.duration = time.time() - self.created_at
        self.state_after = state


class PipelineMemoryStore(PipelineListener):
    def __init__(self):
        self.logs = []

    async def on_transform_enter(self, transform: BaseTransform, state: RAGState):
        self.logs.append(TransformLog(name=transform.name, state_before=state))

    async def on_transform_exit(self, transform: BaseTransform, state: RAGState):
        self.logs[-1].finish(state)

    async def on_enter(self, *args, **kwargs):
        self.logs = []

    async def on_exit(self, *args, **kwargs):
        pass
