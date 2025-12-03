from pydantic import BaseModel
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ContextRecommenderGraphInput(LowerCamelAliasModel):
    smiles: str
    reagents: list[str] = None
    n_conditions: int = 10


class ContextRecommenderGraphOutput(BaseModel):
    __root__: list[dict]


class ContextRecommenderGraphResponse(BaseResponse):
    result: list[dict]


# FP model condition
@register_wrapper(
    name="context_recommender_pr_graph",
    input_class=ContextRecommenderGraphInput,
    output_class=ContextRecommenderGraphOutput,
    response_class=ContextRecommenderGraphResponse
)
class ContextRecommenderWrapper(BaseWrapper):
    """Wrapper class for Context Recommender Predict GRAPH"""
    prefixes = ["context_recommender/v2/predict/GRAPH"]

    def call_raw(self, input: ContextRecommenderGraphInput
                 ) -> ContextRecommenderGraphOutput:
        response = self.session_sync.post(
            f"{self.prediction_url}/api/v2/predict/GRAPH",
            json=input.dict(),
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = ContextRecommenderGraphOutput(__root__=output)

        return output

    def call_sync(self, input: ContextRecommenderGraphInput
                  ) -> ContextRecommenderGraphResponse:
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: ContextRecommenderGraphInput, priority: int = 0
                         ) -> str:
        from askcos2_celery.tasks import context_recommender_task
        async_result = context_recommender_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> ContextRecommenderGraphResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: ContextRecommenderGraphOutput
                                   ) -> ContextRecommenderGraphResponse:
        response = {
            "status_code": 200,
            "message": "",
            "result": output.__root__
        }
        response = ContextRecommenderGraphResponse(**response)

        return response
