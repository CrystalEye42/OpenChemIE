import importlib
import os
from pydantic import BaseModel, Field, validator
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class RetroG2SInput(LowerCamelAliasModel):
    model_name: str = Field(
        default="pistachio_23Q3",
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
            "retro_graph2smiles"]["deployment"]["available_model_names"]
        if v not in available_model_names:
            raise ValueError(f"Unsupported model_name {v} for graph2smiles")

        return v


class RetroG2SResult(BaseModel):
    reactants: list[str] = Field(alias="products")
    scores: list[float]


class RetroG2SOutput(BaseModel):
    __root__: list[RetroG2SResult]


class RetroG2SResponse(BaseResponse):
    result: list[RetroG2SResult]


@register_wrapper(
    name="retro_graph2smiles",
    input_class=RetroG2SInput,
    output_class=RetroG2SOutput,
    response_class=RetroG2SResponse
)
class RetroG2SWrapper(BaseWrapper):
    """Wrapper class for Retro Prediction with Graph2SMILES"""
    prefixes = ["retro/graph2smiles"]

    def call_raw(self, input: RetroG2SInput) -> RetroG2SOutput:
        input_as_dict = input.dict()
        model_name = input_as_dict["model_name"]

        response = self.session_sync.post(
            f"{self.prediction_url}/{model_name}",
            json=input_as_dict,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = RetroG2SOutput(__root__=output)

        return output

    def call_sync(self, input: RetroG2SInput) -> RetroG2SResponse:
        """
        Endpoint for synchronous call to one-step retrosynthesis
        based on Graph2SMILES. https://pubs.acs.org/doi/10.1021/acs.jcim.2c00321
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: RetroG2SInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to one-step retrosynthesis
        based on Graph2SMILES. https://pubs.acs.org/doi/10.1021/acs.jcim.2c00321
        """
        from askcos2_celery.tasks import retro_task
        async_result = retro_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> RetroG2SResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: RetroG2SOutput
                                   ) -> RetroG2SResponse:
        response = {
            "status_code": 200,
            "message": "",
            "result": output.__root__
        }
        response = RetroG2SResponse(**response)

        return response
