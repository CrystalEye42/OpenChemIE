from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Literal
from wrappers import register_wrapper
from wrappers.base import BaseWrapper, BaseResponse


class MolecularComplexityInput(LowerCamelAliasModel):

    smiles: str = Field(
        description="SMILES string of the molecule",
        example="O=C(O)c1ccccc1"
    )

    complexity_metrics: list[str] = Field(
        description="List of complexity metrics to calculate",
        example=["balan", "bertz"]
    )

class MolecularComplexityOutput(BaseModel):

    error: str
    status: str
    results: dict

class MolecularComplexityResponse(BaseResponse):
    result: dict | None


@register_wrapper(
    name="molecular_complexity",
    input_class=MolecularComplexityInput,
    output_class=MolecularComplexityOutput,
    response_class=MolecularComplexityResponse
)
class MolecularComplexityWrapper(BaseWrapper):
    """Wrapper class for Molecular Complexity service"""
    prefixes = ["molecular_complexity"]

    def call_sync(self, input: MolecularComplexityInput
                  ) -> MolecularComplexityResponse:
        """
        Endpoint for synchronous call to the molecular complexity service.
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: MolecularComplexityInput, priority: int = 0
                         ) -> str:
        """
        Endpoint for asynchronous call to the molecular complexity service.
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> MolecularComplexityResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: MolecularComplexityOutput
                                   ) -> MolecularComplexityResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in molecular_complexity " \
                      f"with the following error message {output.error}"
            result = None

        response = MolecularComplexityResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
