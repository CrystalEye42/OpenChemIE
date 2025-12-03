import importlib
import os
from pydantic import BaseModel, Field, validator
from schemas.base import LowerCamelAliasModel
from typing import Any
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class RetroTemplEnumInput(LowerCamelAliasModel):
    smiles: list[str] = Field(
        description="list of target SMILES",
        example=["CN(C)CCOC(c1ccccc1)c1ccccc1"]
    )
    model_name: str | None = Field(
        default="USPTO_50k",
        description="reaction set to be queried against",
    )

    @validator("model_name")
    def check_reaction_set(cls, v, values):
        default_path = "configs.module_config_full"
        config_path = os.environ.get(
            "MODULE_CONFIG_PATH", default_path
        ).replace("/", ".").rstrip(".py")
        module_config = importlib.import_module(config_path).module_config

        available_model_names = module_config[
            "retro_template_enumeration"]["deployment"]["available_model_names"]
        if v is not None and v not in available_model_names:
            raise ValueError(f"Unsupported model_name {v} for template_enumeration")

        return v


class RetroTemplEnumResult(BaseModel):
    reactants: list[str]
    templates: list[dict[str, Any]]
    scores: list[float]


class RetroTemplEnumOutput(BaseModel):
    status: str
    error: str
    results: list[RetroTemplEnumResult]


class RetroTemplEnumResponse(BaseResponse):
    result: list[RetroTemplEnumResult]


@register_wrapper(
    name="retro_template_enumeration",
    input_class=RetroTemplEnumInput,
    output_class=RetroTemplEnumOutput,
    response_class=RetroTemplEnumResponse
)
class RetroTemplEnumWrapper(BaseWrapper):
    """Wrapper class for Retro Prediction with Template Enumeration"""
    prefixes = ["retro/template_enumeration"]

    def call_raw(self, input: RetroTemplEnumInput) -> RetroTemplEnumOutput:
        input_as_dict = input.dict()

        response = self.session_sync.post(
            f"{self.prediction_url}",
            json=input_as_dict,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = RetroTemplEnumOutput(**output)

        return output

    def call_sync(self, input: RetroTemplEnumInput) -> RetroTemplEnumResponse:
        """
        Endpoint for synchronous call to one-step retrosynthesis
        using template enumeration
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: RetroTemplEnumInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to one-step retrosynthesis
        using template enumeration
        """
        from askcos2_celery.tasks import retro_task
        async_result = retro_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> RetroTemplEnumResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: RetroTemplEnumOutput
                                   ) -> RetroTemplEnumResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in retro_template_enumeration " \
                      f"with the following error message {output.error}"
            result = None

        response = RetroTemplEnumResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
