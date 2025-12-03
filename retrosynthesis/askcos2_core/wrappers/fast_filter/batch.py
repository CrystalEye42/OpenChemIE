from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class FastFilterBatchInput(LowerCamelAliasModel):
    rxn_smiles: list[str] = Field(description="list of reaction SMILES")

    class Config:
        schema_extra = {
            "example": {
                "rxn_smiles": [
                    "CC.CC>>CCCC",
                    "CCCC.CC>>CCC",
                    "CCCC.CC>>CCCCCC"
                ]
            }
        }


class FastFilterBatchOutput(BaseModel):
    error: str
    status: str
    results: list[float]


class FastFilterBatchResponse(BaseResponse):
    result: list[float] | None


@register_wrapper(
    name="fast_filter_batch",
    input_class=FastFilterBatchInput,
    output_class=FastFilterBatchOutput,
    response_class=FastFilterBatchResponse
)
class FastFilterBatchWrapper(BaseWrapper):
    """Wrapper class for Fast Filter"""
    prefixes = ["fast_filter/batch"]

    def call_raw(self, input: FastFilterBatchInput) -> FastFilterBatchOutput:
        response = self.session_sync.post(
            f"{self.prediction_url}/fast_filter_evaluate_batch",
            json=input.dict(),
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = FastFilterBatchOutput(**output)

        return output

    def call_sync(self, input: FastFilterBatchInput) -> FastFilterBatchResponse:
        """
        Endpoint for synchronous call to the fast filter, with batch of reactions.
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: FastFilterBatchInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to the fast filter, with batch of reactions.
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> FastFilterBatchResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: FastFilterBatchOutput
                                   ) -> FastFilterBatchResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in fast_filter_batch " \
                      f"with the following error message {output.error}"
            result = None

        response = FastFilterBatchResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
