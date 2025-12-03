from adapters import register_adapter
from pydantic import BaseModel, Field
from wrappers.registry import get_wrapper_registry
from wrappers.scscore.default import (
    SCScoreInput,
    SCScoreResponse
)


class V1SCScoreInput(BaseModel):
    smiles: str = Field(
        description="SMILES string of target"
    )


class V1SCScoreAsyncReturn(BaseModel):
    request: V1SCScoreInput
    score: float


class V1SCScoreResult(BaseModel):
    request: V1SCScoreInput | None
    score: float


@register_adapter(
    name="v1_scscore",
    input_class=V1SCScoreInput,
    result_class=V1SCScoreResult
)
class V1SCScoreAdapter:
    """V1 SCScore Adapter"""
    prefix = "legacy/scscore"
    methods = ["POST"]

    def call_sync(self, input: V1SCScoreInput) -> V1SCScoreResult:
        wrapper = get_wrapper_registry().get_wrapper(module="scscore")
        wrapper_input = self.convert_input(input=input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)
        response = V1SCScoreResult(request=wrapper_input, score=response.score)
        return response

    def __call__(self, input: V1SCScoreInput) -> V1SCScoreResult:
        return self.call_sync(input)

    @staticmethod
    def convert_input(input: V1SCScoreInput) -> SCScoreInput:
        wrapper_input = SCScoreInput(smiles=input.smiles)
        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: SCScoreResponse
                         ) -> V1SCScoreResult:
        result = wrapper_response.result
        result = V1SCScoreResult(score=result)

        return result
