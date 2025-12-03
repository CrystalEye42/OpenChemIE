from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class AtomMapRXNMapperInput(LowerCamelAliasModel):
    smiles: list[str] = Field(description="list of SMILES strings to be atom-mapped")

    class Config:
        schema_extra = {
            "example": {
                "smiles": ["CC.CC>>CCCC", "c1ccccc1>>C1CCCCC1"]
            }
        }


class RXNMapperResult(BaseModel):
    mapped_rxn: str
    confidence: float


class AtomMapRXNMapperOutput(BaseModel):
    error: str
    status: str
    results: list[list[RXNMapperResult | None]]


class AtomMapRXNMapperResponse(BaseResponse):
    result: list[RXNMapperResult | None] | None


@register_wrapper(
    name="atom_map_rxnmapper",
    input_class=AtomMapRXNMapperInput,
    output_class=AtomMapRXNMapperOutput,
    response_class=AtomMapRXNMapperResponse
)
class AtomMapRXNMapperWrapper(BaseWrapper):
    """Wrapper class for IBM RXNMapper"""
    prefixes = ["atom_map/rxnmapper"]

    def call_sync(self, input: AtomMapRXNMapperInput) -> AtomMapRXNMapperResponse:
        """
        Endpoint for synchronous call to IBM RXNMapper, based on
        https://github.com/rxn4chemistry/rxnmapper
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: AtomMapRXNMapperInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to IBM RXNMapper, based on
        https://github.com/rxn4chemistry/rxnmapper
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> AtomMapRXNMapperResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: AtomMapRXNMapperOutput
                                   ) -> AtomMapRXNMapperResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results[0]
        else:
            status_code = 500
            message = f"Backend error encountered in atom_map/rxnmapper " \
                      f"with the following error message {output.error}"
            result = None

        response = AtomMapRXNMapperResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
