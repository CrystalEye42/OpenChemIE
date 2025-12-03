from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class SCScoreInput(LowerCamelAliasModel):
    smiles: str = Field(
        description="SMILES for SCScore calculation. "
                    "Can be multi-molecule (separated by .), in which case "
                    "the max SCScore among the molecules will be returned",
        example="c1ccccc1"
    )


class SCScoreOutput(BaseModel):
    error: str
    status: str
    results: float


class SCScoreResponse(BaseResponse):
    result: float | None


@register_wrapper(
    name="scscore",
    input_class=SCScoreInput,
    output_class=SCScoreOutput,
    response_class=SCScoreResponse
)
class SCScoreWrapper(BaseWrapper):
    """Wrapper class for SCScore"""
    prefixes = ["scscore"]

    def call_sync(self, input: SCScoreInput) -> SCScoreResponse:
        """
        Endpoint for synchronous call to SCScorer.
        https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: SCScoreInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to SCScorer.
        https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> SCScoreResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: SCScoreOutput
                                   ) -> SCScoreResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in scscore " \
                      f"with the following error message {output.error}"
            result = None

        response = SCScoreResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
