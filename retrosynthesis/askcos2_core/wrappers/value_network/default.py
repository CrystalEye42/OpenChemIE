import importlib
import os
from pydantic import BaseModel, Field, validator
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ValueNetworkInput(LowerCamelAliasModel):
    model_name: str = Field(
        default="USPTO_FULL",
        description="model name for torchserve backend"
    )
    smiles: list[str] = Field(
        description="list of target SMILES",
        example=["CS(=N)(=O)Cc1cccc(Br)c1", "CN(C)CCOC(c1ccccc1)c1ccccc1"]
    )

    @validator("model_name")
    def check_model_name(cls, v, values):
        default_path = "configs.module_config_full"
        config_path = os.environ.get(
            "MODULE_CONFIG_PATH", default_path
        ).replace("/", ".").rstrip(".py")
        module_config = importlib.import_module(config_path).module_config

        available_model_names = module_config[
            "value_network"]["deployment"]["available_model_names"]
        if v not in available_model_names:
            raise ValueError(f"Unsupported model_name {v} for value_network")

        return v


class ValueNetworkResult(BaseModel):
    __root__: list[float]


class ValueNetworkOutput(BaseModel):
    __root__: ValueNetworkResult


class ValueNetworkResponse(BaseResponse):
    result: list[ValueNetworkResult]


@register_wrapper(
    name="value_network",
    input_class=ValueNetworkInput,
    output_class=ValueNetworkOutput,
    response_class=ValueNetworkResponse
)
class ValueNetworkWrapper(BaseWrapper):
    """Wrapper class for Retro Prediction with Augmented Transformer"""
    prefixes = ["value_network"]

    def call_raw(self, input: ValueNetworkInput) -> ValueNetworkOutput:
        input_as_dict = input.dict()
        model_name = input_as_dict["model_name"]

        response = self.session_sync.post(
            f"{self.prediction_url}/{model_name}",
            json=input_as_dict,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = list(output)
        return output

    def call_sync(self, input: ValueNetworkInput) -> ValueNetworkResponse:
        """
        Endpoint for synchronous call to Value Network
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: ValueNetworkInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to Value Network
        """
        from askcos2_celery.tasks import retro_task
        async_result = retro_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> ValueNetworkResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: ValueNetworkOutput
                                   ) -> ValueNetworkResponse:
        response = {
            "status_code": 200,
            "message": "",
            "result": [output]
        }
        response = ValueNetworkResponse(**response)

        return response
