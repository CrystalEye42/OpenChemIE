from pydantic import BaseModel
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ContextRecommenderInput(LowerCamelAliasModel):
    smiles: str
    reagents: list[str]
    n_conditions: int = 10


class ContextRecommenderResult(BaseModel):
    agents: list
    temperature: float
    score: float


class ContextRecommenderOutput(BaseModel):
    __root__: list[ContextRecommenderResult]


class ContextRecommenderResponse(BaseResponse):
    result: list[ContextRecommenderResult]


# FP model condition
@register_wrapper(
    name="context_recommender_fp",
    input_class=ContextRecommenderInput,
    output_class=ContextRecommenderOutput,
    response_class=ContextRecommenderResponse
)
class ContextRecommenderWrapper(BaseWrapper):
    """Wrapper class for Context Recommender Condition FP"""
    prefixes = ["context_recommender/v2/condition/FP"]

    def call_raw(self, input: ContextRecommenderInput) -> ContextRecommenderOutput:
        response = self.session_sync.post(
            f"{self.prediction_url}/api/v2/condition/FP",
            json=input.dict(),
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = ContextRecommenderOutput(__root__=output)

        return output

    def call_sync(self, input: ContextRecommenderInput) -> ContextRecommenderResponse:
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: ContextRecommenderInput, priority: int = 0
                         ) -> str:
        from askcos2_celery.tasks import context_recommender_task
        async_result = context_recommender_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> ContextRecommenderResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: ContextRecommenderOutput
                                   ) -> ContextRecommenderResponse:
        response = {
            "status_code": 200,
            "message": "",
            "result": output.__root__
        }
        response = ContextRecommenderResponse(**response)

        return response
