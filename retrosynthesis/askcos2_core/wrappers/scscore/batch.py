from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class SCScoreBatchInput(LowerCamelAliasModel):
    smiles: list[str] = Field(
        description="list of SMILES for SCScore calculation. "
                    "Can be multi-molecule (separated by .), in which case "
                    "the max SCScore among the molecules will be returned",
        example=["c1ccccc1", "CCCBr"]
    )


class SCScoreBatchOutput(BaseModel):
    error: str
    status: str
    results: list[float]


class SCScoreBatchResponse(BaseResponse):
    result: dict[str, float] | None


@register_wrapper(
    name="scscore_batch",
    input_class=SCScoreBatchInput,
    output_class=SCScoreBatchOutput,
    response_class=SCScoreBatchResponse
)
class SCScoreBatchWrapper(BaseWrapper):
    """Wrapper class for SCScore"""
    prefixes = ["scscore/batch"]

    def call_raw(self, input: SCScoreBatchInput) -> SCScoreBatchOutput:
        json = input.dict()
        # aliasing, for backward compatibility (with frontend calls)
        json["smiles_list"] = json.pop("smiles")

        response = self.session_sync.post(
            f"{self.prediction_url}_batch",
            json=json,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = SCScoreBatchOutput(**output)

        return output

    def call_sync(self, input: SCScoreBatchInput) -> SCScoreBatchResponse:
        """
        Endpoint for synchronous call to batch SCScorer.
        https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622
        """
        output = self.call_raw(input=input)
        results = {
            smi: score
            for smi, score in zip(input.smiles, output.results)
        }
        response = self.convert_results_to_response(results)

        return response

    async def call_async(self, input: SCScoreBatchInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to batch SCScorer.
        https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> SCScoreBatchResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_results_to_response(results: dict[str, float]
                                    ) -> SCScoreBatchResponse:
        response = SCScoreBatchResponse(
            status_code=200,
            message="",
            result=results
        )

        return response
