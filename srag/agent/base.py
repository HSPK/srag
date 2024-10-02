from typing import Any

import pydantic
from modelhub import AsyncModelhub


class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


class AgentCallback(BaseModel):
    async def on_enter(self, locals: dict, globals: dict):  # noqa: A002
        pass

    async def on_exit(self, locals: dict, globals: dict):  # noqa: A002
        pass


class Agent(BaseModel):
    name: str = "agent"
    llm_model: str = "gpt-4o"
    llm_client: Any = None
    llm_default_parameters: dict = {
        "temperature": 0.1,
    }
    input_keys: set = set()
    output_keys: set = set()
    callbacks: list[AgentCallback] = []

    def _init_llm(self):
        if self.llm_client is not None:
            return
        self.llm_client = AsyncModelhub()

    async def llm(self, prompt: str, parameters: dict | None = None, **kwargs):
        self._init_llm()
        if parameters is None:
            parameters = self.llm_default_parameters
        parameters = {**self.llm_default_parameters, **parameters, **kwargs}

        output = await self.llm_client.achat(prompt, parameters=parameters, model=self.llm_model)
        return {"text": output.generated_text, "raw": output}

    async def stream_llm(self, prompt: str, parameters: dict | None = None, **kwargs):
        self._init_llm()
        if parameters is None:
            parameters = self.llm_default_parameters
        parameters = {**self.llm_default_parameters, **parameters, **kwargs}

        async for token in self.llm_client.astream_chat(
            prompt, parameters=parameters, model=self.llm_model
        ):
            yield {"text": token.token.text}

    async def validate_inputs(self, inputs: dict):
        for key in self.input_keys:
            if key not in inputs:
                msg = f"Missing input key: {key}"
                raise ValueError(msg)

    async def _call(self, inputs: dict):
        raise NotImplementedError

    async def __call__(self, inputs: dict):
        for callback in self.callbacks:
            await callback.on_enter(locals(), globals())
        await self.validate_inputs(inputs)
        res = await self._call(inputs)
        for callback in self.callbacks:
            await callback.on_exit(locals(), globals())
        return {key: res[key] for key in self.output_keys}


class PromptAgent(Agent):
    """Agent that use prompt to do task.
    @param prompt: str, the prompt to use.
    @param input_keys: set, the keys of inputs.
    @param output_keys: set, the keys of outputs. implement parse_response to parse the response.
    """

    name: str = "prompt_agent"
    input_keys: set = ("query",)
    output_keys: set = ("raw",)
    prompt: str

    async def form_prompt(self, inputs: dict):
        format_dict = {key: inputs[key] for key in self.input_keys}
        return self.prompt.format(**format_dict)

    async def parse_response(self, response: str):
        raise NotImplementedError

    async def _call(self, inputs: dict):
        prompt = await self.form_prompt(inputs)
        response = (await self.llm(prompt))["text"]
        return await self.parse_response(response)

    async def stream(self, inputs: dict):
        prompt = await self.form_prompt(inputs)
        async for delta in self.stream_llm(prompt):
            yield await self.parse_response(delta["text"])
