from adapters import register_adapter
from pydantic import BaseModel, Field
from wrappers.registry import get_wrapper_registry
from wrappers.scscore.batch import (
    SCScoreBatchInput,
    SCScoreBatchResponse
)


class V1SCScoreBatchInput(BaseModel):
    smiles: list[str] = Field(
        description="list of SMILES for SCScore calculation. "
                    "Can be multi-molecule (separated by .), in which case "
                    "the max SCScore among the molecules will be returned",
        example=["c1ccccc1", "CCCBr"]
    )


class V1SCScoreBatchAsyncReturn(BaseModel):
    request: V1SCScoreBatchInput
    score: float


class V1SCScoreBatchResult(BaseModel):
    request: V1SCScoreBatchInput | None
    result: dict[str, float] | None


@register_adapter(
    name="v1_scscore_batch",
    input_class=V1SCScoreBatchInput,
    result_class=V1SCScoreBatchResult
)
class V1SCScoreBatchAdapter:
    """V1 SCScore Batch Adapter"""
    prefix = "legacy/scscore/batch"
    methods = ["POST"]

    def call_sync(self, input: V1SCScoreBatchInput) -> V1SCScoreBatchResult:
        wrapper = get_wrapper_registry().get_wrapper(module="scscore_batch")
        wrapper_input = self.convert_input(input=input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)
        response = V1SCScoreBatchResult(request=wrapper_input, result=response.result)
        return response

    def __call__(self, input: V1SCScoreBatchInput) -> V1SCScoreBatchResult:
        return self.call_sync(input)

    @staticmethod
    def convert_input(input: V1SCScoreBatchInput) -> SCScoreBatchInput:
        wrapper_input = SCScoreBatchInput(smiles=input.smiles)
        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: SCScoreBatchResponse
                         ) -> V1SCScoreBatchResult:
        result = wrapper_response.result
        print(result)
        result = V1SCScoreBatchResult(result=result)

        return result
