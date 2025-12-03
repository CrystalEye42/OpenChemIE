from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Literal
from wrappers import register_wrapper
from wrappers.base import BaseWrapper, BaseResponse


class MolecularComplexityBatchInput(LowerCamelAliasModel):

    smiles_list: list[str] = Field(
        description="List of SMILES strings of the molecules",
        example=["O=C(O)c1ccccc1", "Brc1ccccc1"]
    )

    complexity_metrics: list[str] = Field(
        description="List of complexity metrics to calculate",
        example=["balan", "bertz"]
    )

class MolecularComplexityBatchOutput(BaseModel):

    error: str
    status: str
    results: list[dict]

class MolecularComplexityBatchResponse(BaseResponse):
    result: list[dict] | None


@register_wrapper(
    name="molecular_complexity_batch",
    input_class=MolecularComplexityBatchInput,
    output_class=MolecularComplexityBatchOutput,
    response_class=MolecularComplexityBatchResponse
)
class MolecularComplexityBatchWrapper(BaseWrapper):
    """Wrapper class for Molecular Complexity service"""
    prefixes = ["molecular_complexity/batch"]

    def call_sync(self, input: MolecularComplexityBatchInput
                  ) -> MolecularComplexityBatchResponse:
        """
        Endpoint for synchronous call to the molecular complexity service.
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: MolecularComplexityBatchInput, priority: int = 0
                         ) -> str:
        """
        Endpoint for asynchronous call to the molecular complexity service.
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> MolecularComplexityBatchResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: MolecularComplexityBatchOutput
                                   ) -> MolecularComplexityBatchResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in molecular_complexity " \
                      f"with the following error message {output.error}"
            result = None

        response = MolecularComplexityBatchResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
