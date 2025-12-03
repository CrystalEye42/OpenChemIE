from adapters import register_adapter
from pydantic import BaseModel, Field
from wrappers.registry import get_wrapper_registry
from wrappers.reaction_classification.default import (
    ReactionClassificationInput,
    ReactionClassificationResult,
    ReactionClassificationResponse
)


class V1ReactionClassificationInput(BaseModel):
    reactants: str = Field(
        description="SMILES string of reactants"
    )
    products: str = Field(
        description="SMILES string of products"
    )


class V1ReactionClassificationAsyncReturn(BaseModel):
    request: V1ReactionClassificationInput
    task_id: str


class V1ReactionClassificationResult(BaseModel):
    status: str
    message: str
    result: list[ReactionClassificationResult]


@register_adapter(
    name="v1_reaction_classification",
    input_class=V1ReactionClassificationInput,
    result_class=V1ReactionClassificationResult
)
class V1ReactionClassificationAdapter:
    """V1 Reaction Classification Adapter"""
    prefix = "legacy/reaction_classification"
    methods = ["POST"]

    def call_sync(self, input: V1ReactionClassificationInput) -> V1ReactionClassificationResult:
        wrapper = get_wrapper_registry().get_wrapper(module="reaction_classification")
        wrapper_input = self.convert_input(input=input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)

        return response

    async def __call__(self, input: V1ReactionClassificationInput, priority: int = 0
                       ) -> V1ReactionClassificationAsyncReturn:
        from askcos2_celery.tasks import legacy_task

        async_result = legacy_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        async_return = V1ReactionClassificationAsyncReturn(
            request=input, task_id=task_id)

        return async_return

    # def __call__(self, input: V1ReactionClassificationInput) -> V1ReactionClassificationResult:
    #     return self.call_sync(input)

    @staticmethod
    def convert_input(input: V1ReactionClassificationInput) -> ReactionClassificationInput:
        wrapper_input = ReactionClassificationInput(smiles=[input.reactants+">>"+input.products])

        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: ReactionClassificationResponse
                         ) -> V1ReactionClassificationResult:
        result = wrapper_response.result
        result = V1ReactionClassificationAdapter.convert_wrapper_result_into_v1_result(result)
        return result

    @staticmethod
    def convert_wrapper_result_into_v1_result(result: list[ReactionClassificationResult]) -> V1ReactionClassificationResult:
        v1_result = {
            'status': 'OK',
            'message': 'OK',
            'result': result
        }
        v1_result = V1ReactionClassificationResult(**v1_result)

        return v1_result