from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class FastFilterInput(LowerCamelAliasModel):
    smiles: list[str] = Field(description="tuple of reactant SMILES and target SMILES")

    class Config:
        schema_extra = {
            "example": {
                "smiles": ["CC.CC", "CCCC"]
            }
        }


class FastFilterOutcome(BaseModel):
    smiles: str
    template_ids: list
    num_examples: int


class FastFilterResult(BaseModel):
    rank: float
    outcome: FastFilterOutcome
    score: float
    prob: float


class FastFilterOutput(BaseModel):
    error: str
    status: str
    results: list[list[list[FastFilterResult]]]


class FastFilterResponse(BaseResponse):
    result: FastFilterResult | None


@register_wrapper(
    name="fast_filter",
    input_class=FastFilterInput,
    output_class=FastFilterOutput,
    response_class=FastFilterResponse
)
class FastFilterWrapper(BaseWrapper):
    """Wrapper class for Fast Filter"""
    prefixes = ["fast_filter"]

    def call_raw(self, input: FastFilterInput) -> FastFilterOutput:
        response = self.session_sync.post(
            f"{self.prediction_url}/fast_filter_evaluate",
            json=input.dict(),
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = FastFilterOutput(**output)

        return output

    def call_sync(self, input: FastFilterInput) -> FastFilterResponse:
        """
        Endpoint for synchronous call to the fast filter.
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: FastFilterInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to the fast filter.
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> FastFilterResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: FastFilterOutput
                                   ) -> FastFilterResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results[0][0][0]
        else:
            status_code = 500
            message = f"Backend error encountered in fast_filter " \
                      f"with the following error message {output.error}"
            result = None

        response = FastFilterResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
