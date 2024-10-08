from srag.pipeline import TransformListener
from srag.pipeline.vanilla import Generation

from ..pipeline import BaseTransform


class Agent(BaseTransform):
    def __init__(
        self,
        transforms=None,
        run_in_parallel=False,
        run_type="ignore",
        input_key=None,
        output_key=None,
        shared=None,
        listeners: list[TransformListener] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            transforms, run_in_parallel, run_type, input_key, output_key, shared, *args, **kwargs
        )
        if listeners:
            self.shared = self.shared or self._default_sharedresource()
            self.shared.listener.listeners = listeners


class PromptAgent(Agent):
    def __init__(
        self,
        llm_model: str,
        input_key: list[str],
        output_key: list[str],
        *args,
        **kwargs,
    ):
        super().__init__(input_key=input_key, output_key=output_key, *args, **kwargs)
        self.genration = Generation(llm_model=llm_model)
        self.prompt = "\n".join([x.strip() for x in self.__doc__.split("\n")])

    async def form_prompt(self, inputs: dict):
        format_dict = {key: inputs.get(key) for key in self.input_key}
        return self.prompt.format(**format_dict)

    async def parse_response(self, response: str):
        raise NotImplementedError

    async def transform(self, state, **kwargs):
        final_prompt = await self.form_prompt(state)
        state["final_prompt"] = final_prompt
        state = await self.genration(state)
        output = await self.parse_response(state["response"])
        state["parsed_response"] = output
        return state

    async def stream_transform(self, state, **kwargs):
        final_prompt = await self.form_prompt(state)
        state["final_prompt"] = final_prompt
        async for state in self.genration.stream(state):
            yield state
        output = await self.parse_response(state["response"])
        state["parsed_response"] = output
        yield state


class ChatAgent(PromptAgent):
    """User question: {query}
    Your answer:"""

    def __init__(
        self,
        llm_model,
        input_key: list[str] = ["query"],
        output_key: list[str] = [],
        *args,
        **kwargs,
    ):
        super().__init__(llm_model, input_key, output_key, *args, **kwargs)

    async def parse_response(self, response):
        return {}
