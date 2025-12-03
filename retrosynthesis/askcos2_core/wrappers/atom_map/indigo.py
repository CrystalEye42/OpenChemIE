from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class AtomMapIndigoInput(LowerCamelAliasModel):
    smiles: list[str] = Field(description="list of SMILES strings to be atom-mapped")

    class Config:
        schema_extra = {
            "example": {
                "smiles": ["CC.CC>>CCCC", "c1ccccc1>>C1CCCCC1"]
            }
        }


class AtomMapIndigoOutput(BaseModel):
    error: str
    status: str
    results: list[list[str]]


class AtomMapIndigoResponse(BaseResponse):
    result: list[str] | None


@register_wrapper(
    name="atom_map_indigo",
    input_class=AtomMapIndigoInput,
    output_class=AtomMapIndigoOutput,
    response_class=AtomMapIndigoResponse
)
class AtomMapIndigoWrapper(BaseWrapper):
    """Wrapper class for Indigo Atom Mapper"""
    prefixes = ["atom_map/indigo"]

    def call_sync(self, input: AtomMapIndigoInput) -> AtomMapIndigoResponse:
        """
        Endpoint for synchronous call to indigo atom mapper, based on
        https://lifescience.opensource.epam.com/indigo/api/index.html
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: AtomMapIndigoInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to indigo atom mapper, based on
        https://lifescience.opensource.epam.com/indigo/api/index.html
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> AtomMapIndigoResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: AtomMapIndigoOutput
                                   ) -> AtomMapIndigoResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results[0]
        else:
            status_code = 500
            message = f"Backend error encountered in atom_map/indigo " \
                      f"with the following error message {output.error}"
            result = None

        response = AtomMapIndigoResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
