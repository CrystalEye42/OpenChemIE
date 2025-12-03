from adapters import register_adapter
from pydantic import BaseModel, Field
from wrappers.registry import get_wrapper_registry
from wrappers.fast_filter.default import (
    FastFilterInput,
    FastFilterResponse
)


class V1FastFilterInput(BaseModel):
    reactants: str = Field(
        description="SMILES string of reactants"
    )
    products: str = Field(
        description="SMILES string of products"
    )


class V1FastFilterAsyncReturn(BaseModel):
    request: V1FastFilterInput
    task_id: str


class V1FastFilterResult(BaseModel):
    __root__: float


@register_adapter(
    name="v1_fast_filter",
    input_class=V1FastFilterInput,
    result_class=V1FastFilterResult
)
class V1FastFilterAdapter:
    """V1 FastFilter Adapter"""
    prefix = "legacy/fast_filter"
    methods = ["POST"]

    def call_sync(self, input: V1FastFilterInput) -> V1FastFilterResult:
        wrapper = get_wrapper_registry().get_wrapper(module="fast_filter")
        wrapper_input = self.convert_input(input=input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)

        return response

    async def __call__(self, input: V1FastFilterInput, priority: int = 0
                       ) -> V1FastFilterAsyncReturn:
        from askcos2_celery.tasks import legacy_task

        async_result = legacy_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        async_return = V1FastFilterAsyncReturn(
            request=input, task_id=task_id)

        return async_return

    @staticmethod
    def convert_input(input: V1FastFilterInput) -> FastFilterInput:
        wrapper_input = FastFilterInput(
            smiles=[input.reactants, input.products]
        )

        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: FastFilterResponse
                         ) -> V1FastFilterResult:
        result = wrapper_response.result.score
        result = V1FastFilterResult(__root__=result)

        return result
