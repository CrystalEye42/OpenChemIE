from adapters import register_adapter
from pydantic import BaseModel, Field
from wrappers.registry import get_wrapper_registry
from wrappers.site_selectivity.default import (
    SiteSelectivityInput,
    SiteSelectivityResult,
    SiteSelectivityResponse
)


class V1SelectivityInput(BaseModel):
    smiles: str = Field(
        description="SMILES string of target"
    )


class V1SelectivityAsyncReturn(BaseModel):
    request: V1SelectivityInput
    task_id: str


class V1SelectivityResult(BaseModel):
    __root__: list[SiteSelectivityResult]


@register_adapter(
    name="v1_selectivity",
    input_class=V1SelectivityInput,
    result_class=V1SelectivityResult
)
class V1SelectivityAdapter:
    """V1 Site Selectivity Adapter"""
    prefix = "legacy/selectivity"
    methods = ["POST"]

    def call_sync(self, input: V1SelectivityInput) -> V1SelectivityResult:
        wrapper = get_wrapper_registry().get_wrapper(module="site_selectivity")
        wrapper_input = self.convert_input(input=input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)

        return response

    async def __call__(self, input: V1SelectivityInput, priority: int = 0
                       ) -> V1SelectivityAsyncReturn:
        from askcos2_celery.tasks import legacy_task

        async_result = legacy_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        async_return = V1SelectivityAsyncReturn(
            request=input, task_id=task_id)

        return async_return

    @staticmethod
    def convert_input(input: V1SelectivityInput) -> SiteSelectivityInput:
        wrapper_input = SiteSelectivityInput(smiles=[input.smiles])

        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: SiteSelectivityResponse
                         ) -> V1SelectivityResult:
        result = wrapper_response.result[0]
        result = V1SelectivityResult(__root__=result)

        return result
