import importlib
import os
from pydantic import BaseModel, Field, validator
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class RetroExactMatchInput(LowerCamelAliasModel):
    smiles: list[str] = Field(
        description="list of target SMILES",
        example=["CN(C)CCOC(c1ccccc1)c1ccccc1"]
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
            "retro_exact_match"]["deployment"]["available_model_names"]
        if v is not None and v not in available_model_names:
            raise ValueError(f"Unsupported reaction_set {v} for exact_match")

        return v


class RetroExactMatchResult(BaseModel):
    reactants: list[str]
    scores: list[float]
    reaction_ids: list[str]
    reaction_sets: list[str]
    reaction_data: list[dict]


class RetroExactMatchOutput(BaseModel):
    status: str
    error: str
    results: list[RetroExactMatchResult]


class RetroExactMatchResponse(BaseResponse):
    result: list[RetroExactMatchResult]


@register_wrapper(
    name="retro_exact_match",
    input_class=RetroExactMatchInput,
    output_class=RetroExactMatchOutput,
    response_class=RetroExactMatchResponse
)
class RetroExactMatchWrapper(BaseWrapper):
    """Wrapper class for Retro Prediction with Exact Match"""
    prefixes = ["retro/exact_match"]

    def call_raw(self, input: RetroExactMatchInput) -> RetroExactMatchOutput:
        input_as_dict = input.dict()

        response = self.session_sync.post(
            f"{self.prediction_url}",
            json=input_as_dict,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = RetroExactMatchOutput(**output)

        return output

    def call_sync(self, input: RetroExactMatchInput) -> RetroExactMatchResponse:
        """
        Endpoint for synchronous call to one-step retrosynthesis using exact match
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: RetroExactMatchInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to one-step retrosynthesis using exact match
        """
        from askcos2_celery.tasks import retro_task
        async_result = retro_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> RetroExactMatchResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: RetroExactMatchOutput
                                   ) -> RetroExactMatchResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in retro_exact_match " \
                      f"with the following error message {output.error}"
            result = None

        response = RetroExactMatchResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
