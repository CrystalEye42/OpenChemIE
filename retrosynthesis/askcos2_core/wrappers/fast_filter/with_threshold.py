from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class FastFilterWThresholdInput(LowerCamelAliasModel):
    smiles: list[str] = Field(description="tuple of reactant SMILES and target SMILES")
    threshold: float


class FastFilterWThresholdResult(BaseModel):
    flag: bool
    score: float


class FastFilterWThresholdOutput(BaseModel):
    error: str
    status: str
    results: list[FastFilterWThresholdResult]


class FastFilterWThresholdResponse(BaseResponse):
    result: FastFilterWThresholdResult | None


@register_wrapper(
    name="fast_filter_with_threshold",
    input_class=FastFilterWThresholdInput,
    output_class=FastFilterWThresholdOutput,
    response_class=FastFilterWThresholdResponse
)
class FastFilterWThresholdWrapper(BaseWrapper):
    """Wrapper class for Fast Filter with Threshold"""
    prefixes = ["fast_filter/with_threshold"]

    def call_raw(self, input: FastFilterWThresholdInput) -> FastFilterWThresholdOutput:
        response = self.session_sync.post(
            f"{self.prediction_url}/filter_with_threshold",
            json=input.dict(),
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = FastFilterWThresholdOutput(**output)

        return output

    def call_sync(self, input: FastFilterWThresholdInput
                  ) -> FastFilterWThresholdResponse:
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: FastFilterWThresholdInput, priority: int = 0
                         ) -> str:
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> FastFilterWThresholdResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: FastFilterWThresholdOutput
                                   ) -> FastFilterWThresholdResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results[0]
        else:
            status_code = 500
            message = f"Backend error encountered in fast_filter_with_threshold " \
                      f"with the following error message {output.error}"
            result = None

        response = FastFilterWThresholdResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
