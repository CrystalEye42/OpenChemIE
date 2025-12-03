from pydantic import BaseModel
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class DescriptorsInput(LowerCamelAliasModel):
    smiles: str


class DescriptorsResult(BaseModel):
    smiles: str
    partial_charge: list[float]
    fukui_neu: list[float]
    fukui_elec: list[float]
    NMR: list[float]
    bond_order: list[float]
    bond_length: list[float]


class DescriptorsOutput(BaseModel):
    error: str
    status: str
    results: list[DescriptorsResult]


class DescriptorsResponse(BaseResponse):
    result: DescriptorsResult


@register_wrapper(
    name="descriptors",
    input_class=DescriptorsInput,
    output_class=DescriptorsOutput,
    response_class=DescriptorsResponse
)
class DescriptorWrapper(BaseWrapper):
    """Wrapper class for Descriptors"""
    prefixes = ["descriptors"]

    def call_sync(self, input: DescriptorsInput) -> DescriptorsResponse:
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: DescriptorsInput, priority: int = 0) -> str:
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> DescriptorsResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: DescriptorsOutput
                                   ) -> DescriptorsResponse:
        response = {
            "status_code": 200,
            "message": "",
            "result": output.results[0]
        }
        response = DescriptorsResponse(**response)

        return response
