from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class AtomMapWLNInput(LowerCamelAliasModel):
    smiles: list[str] = Field(description="list of SMILES strings to be atom-mapped")

    class Config:
        schema_extra = {
            "example": {
                "smiles": ["CC.CC>>CCCC", "c1ccccc1>>C1CCCCC1"]
            }
        }


class AtomMapWLNOutput(BaseModel):
    error: str
    status: str
    results: list[str]


class AtomMapWLNResponse(BaseResponse):
    result: list[str] | None


@register_wrapper(
    name="atom_map_wln",
    input_class=AtomMapWLNInput,
    output_class=AtomMapWLNOutput,
    response_class=AtomMapWLNResponse
)
class AtomMapWLNWrapper(BaseWrapper):
    """Wrapper class for WLN Atom Mapper"""
    prefixes = ["atom_map/wln"]

    def call_sync(self, input: AtomMapWLNInput) -> AtomMapWLNResponse:
        """
        Endpoint for synchronous call to WLN-based atom mapper
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: AtomMapWLNInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to WLN-based atom mapper
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> AtomMapWLNResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: AtomMapWLNOutput
                                   ) -> AtomMapWLNResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in atom_map/wln " \
                      f"with the following error message {output.error}"
            result = None

        response = AtomMapWLNResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
