from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Literal
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ClusterInput(LowerCamelAliasModel):
    original: str = Field(
        description="SMILES string or the target molecule",
        example="CCCCCC"
    )
    outcomes: list[str] = Field(
        description="list of SMILES strings of precursors",
        example=["CCC.CCC", "CCCC.CC"]
    )
    feature: Literal["original", "outcomes", "all"] = "original"
    cluster_method: Literal["kmeans", "hdbscan", "rxn_class"] = "rxn_class"
    fp_type: Literal["morgan"] = Field(
        default="morgan",
        description="fingerprint type"
    )
    fp_length: int = Field(
        default=512,
        description="fingerprint bits"
    )
    fp_radius: int = Field(
        default=1,
        description="fingerprint radius"
    )
    scores: list[float] = Field(
        default=None,
        description="list of scores of precursors",
        example=[0.5, 0.6]
    )
    classification_threshold: float = Field(
        default=0.2,
        description="threshold to classify a reaction as unknown when "
                    "clustering by 'rxn_class'"
    )


class ClusterOutput(BaseModel):
    error: str
    status: str
    results: list[list[int] | dict[str, str]]


class ClusterResponse(BaseResponse):
    result: list[list[int] | dict[str, str]] | None


@register_wrapper(
    name="cluster",
    input_class=ClusterInput,
    output_class=ClusterOutput,
    response_class=ClusterResponse
)
class ClusterWrapper(BaseWrapper):
    """Wrapper class for clustering reactions"""
    prefixes = ["cluster"]

    def call_sync(self, input: ClusterInput
                  ) -> ClusterResponse:
        """
        Endpoint for synchronous call to the reaction clustering service
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: ClusterInput, priority: int = 0
                         ) -> str:
        """
        Endpoint for asynchronous call to the reaction clustering service
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> ClusterResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: ClusterOutput
                                   ) -> ClusterResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in cluster " \
                      f"with the following error message {output.error}"
            result = None

        response = ClusterResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
