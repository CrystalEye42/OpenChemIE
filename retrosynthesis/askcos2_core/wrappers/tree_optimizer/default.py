from pydantic import BaseModel
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseWrapper, BaseResponse


class TreeOptimizerInput(LowerCamelAliasModel):
    targets: list[str]
    graph: dict


class TreeOptimizerResult(BaseModel):
    paths: list[dict]


class TreeOptimizerOutput(BaseModel):
    error: str
    status: str
    results: TreeOptimizerResult


class TreeOptimizerResponse(BaseResponse):
    result: TreeOptimizerResult


@register_wrapper(
    name="tree_optimizer",
    input_class=TreeOptimizerInput,
    output_class=TreeOptimizerOutput,
    response_class=TreeOptimizerResponse
)
class TreeOptimizerWrapper(BaseWrapper):
    """Wrapper class for Tree Optimizer"""
    prefixes = ["tree_optimizer"]

    def call_sync(self, input: TreeOptimizerInput
                  ) -> TreeOptimizerResponse:
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: TreeOptimizerInput, priority: int = 0) -> str:
        from askcos2_celery.tasks import tree_optimizer_task
        async_result = tree_optimizer_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> TreeOptimizerResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: TreeOptimizerOutput
                                   ) -> TreeOptimizerResponse:
        response = {
            "status_code": 200,
            "message": "",
            "result": output.results
        }
        response = TreeOptimizerResponse(**response)

        return response
