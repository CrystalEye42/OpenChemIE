import importlib
import os
from pydantic import BaseModel, Field, validator
from schemas.base import LowerCamelAliasModel
from typing import Any, Literal
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class AttributeFilter(BaseModel):
    name: str
    logic: Literal[">", ">=", "<", "<=", "=="]
    value: int | float


class RetroTemplRelInput(LowerCamelAliasModel):
    model_name: str = Field(
        default="reaxys",
        description="model name for torchserve backend"
    )
    smiles: list[str] = Field(
        description="list of target SMILES",
        example=["CS(=N)(=O)Cc1cccc(Br)c1", "CN(C)CCOC(c1ccccc1)c1ccccc1"]
    )
    max_num_templates: int = Field(
        default=1000,
        description="number of templates to consider"
    )
    max_cum_prob: float = Field(
        default=0.995,
        description="maximum cumulative probability of templates"
    )
    attribute_filter: list[AttributeFilter] = Field(
        default_factory=list,
        description="template attribute filter to apply before template application",
        example=[]
    )

    @validator("model_name")
    def check_model_name(cls, v, values):
        default_path = "configs.module_config_full"
        config_path = os.environ.get(
            "MODULE_CONFIG_PATH", default_path
        ).replace("/", ".").rstrip(".py")
        module_config = importlib.import_module(config_path).module_config

        available_model_names = module_config[
            "retro_template_relevance"]["deployment"]["available_model_names"]
        if v not in available_model_names:
            raise ValueError(f"Unsupported model_name {v} for template_relevance")

        return v


class RetroTemplRelResult(BaseModel):
    templates: list[dict[str, Any]]
    reactants: list[str]
    scores: list[float]


class RetroTemplRelOutput(BaseModel):
    __root__: list[RetroTemplRelResult]


class RetroTemplRelResponse(BaseResponse):
    result: list[RetroTemplRelResult]


@register_wrapper(
    name="retro_template_relevance",
    input_class=RetroTemplRelInput,
    output_class=RetroTemplRelOutput,
    response_class=RetroTemplRelResponse
)
class RetroTemplRelWrapper(BaseWrapper):
    """Wrapper class for One-step Retrosynthesis with Template Relevance Model"""
    prefixes = ["retro/template_relevance"]

    def call_raw(self, input: RetroTemplRelInput) -> RetroTemplRelOutput:
        input_as_dict = input.dict()
        model_name = input_as_dict["model_name"]

        response = self.session_sync.post(
            f"{self.prediction_url}/{model_name}",
            json=input_as_dict,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = RetroTemplRelOutput(__root__=output)

        return output

    def call_sync(self, input: RetroTemplRelInput) -> RetroTemplRelResponse:
        """
        Endpoint for synchronous call to template-relevance one-step retrosynthesis,
        a variant of NeuralSym.
        https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/chem.201605499
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: RetroTemplRelInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to template-relevance one-step retrosynthesis,
        a variant of NeuralSym.
        https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/chem.201605499
        """
        from askcos2_celery.tasks import retro_task
        async_result = retro_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> RetroTemplRelResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: RetroTemplRelOutput
                                   ) -> RetroTemplRelResponse:
        response = {
            "status_code": 200,
            "message": "",
            "result": output.__root__
        }
        response = RetroTemplRelResponse(**response)

        return response
