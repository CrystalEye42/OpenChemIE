import importlib
import os
from pydantic import BaseModel, Field, validator
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class RetroRSimInput(LowerCamelAliasModel):
    smiles: list[str] = Field(
        description="list of target SMILES",
        example=["CS(=N)(=O)Cc1cccc(Br)c1", "CN(C)CCOC(c1ccccc1)c1ccccc1"]
    )
    threshold: float = Field(
        description="threshold for similarity",
        example=0.3
    )
    top_k: int = Field(
        description="return top k results",
        example=10
    )
    reaction_set: str | None = Field(
        default="USPTO_FULL",
        description="reaction set to be queried against",
    )

    @validator("reaction_set")
    def check_reaction_set(cls, v, values):
        default_path = "configs.module_config_full"
        config_path = os.environ.get(
            "MODULE_CONFIG_PATH", default_path
        ).replace("/", ".").rstrip(".py")
        module_config = importlib.import_module(config_path).module_config

        available_model_names = module_config[
            "retro_retrosim"]["deployment"]["available_model_names"]
        if v is not None and v not in available_model_names:
            raise ValueError(f"Unsupported reaction_set {v} for retrosim")

        return v


class RetroRSimResult(BaseModel):
    reactants: list[str] = Field(alias="products")
    scores: list[float]
    reaction_ids: list[str]
    reaction_sets: list[str]
    reaction_data: list[dict]


class RetroRSimOutput(BaseModel):
    status: str
    error: str
    results: list[RetroRSimResult]


class RetroRSimResponse(BaseResponse):
    result: list[RetroRSimResult]


@register_wrapper(
    name="retro_retrosim",
    input_class=RetroRSimInput,
    output_class=RetroRSimOutput,
    response_class=RetroRSimResponse
)
class RetroRSimWrapper(BaseWrapper):
    """Wrapper class for Retro Prediction with Retrosim"""
    prefixes = ["retro/retrosim"]

    def call_raw(self, input: RetroRSimInput) -> RetroRSimOutput:
        input_as_dict = input.dict()

        response = self.session_sync.post(
            f"{self.prediction_url}",
            json=input_as_dict,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = RetroRSimOutput(**output)

        return output

    def call_sync(self, input: RetroRSimInput) -> RetroRSimResponse:
        """
        Endpoint for synchronous call to one-step retrosynthesis based on
        RetroSim. https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00355
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: RetroRSimInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to one-step retrosynthesis based on
        Retrosim. https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00355
        """
        from askcos2_celery.tasks import retro_task
        async_result = retro_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> RetroRSimResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: RetroRSimOutput
                                   ) -> RetroRSimResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in retrosim " \
                      f"with the following error message {output.error}"
            result = None

        response = RetroRSimResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
