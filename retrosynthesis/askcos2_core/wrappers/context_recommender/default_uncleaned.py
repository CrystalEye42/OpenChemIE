from pydantic import BaseModel
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ContextRecommenderInput(LowerCamelAliasModel):
    smiles: str
    n_conditions: int = 10
    with_smiles: bool = False
    return_scores: bool = True


class ContextRecommenderOutput(BaseModel):
    conditions: list[list]
    scores: list[float] | None


class ContextRecommenderResponse(BaseResponse):
    result: ContextRecommenderOutput


# FP model condition
@register_wrapper(
    name="context_recommender_uncleaned",
    input_class=ContextRecommenderInput,
    output_class=ContextRecommenderOutput,
    response_class=ContextRecommenderResponse
)
class ContextRecommenderWrapper(BaseWrapper):
    """Wrapper class for Context Recommender uncleaned output"""
    prefixes = ["context_recommender/v1/condition_uncleaned"]

    def call_raw(self, input: ContextRecommenderInput) -> ContextRecommenderOutput:
        response = self.session_sync.post(
            f"{self.prediction_url}/api/v1/condition_uncleaned",
            json=input.dict(),
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = ContextRecommenderOutput(**output)

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
            "result": output
        }
        response = ContextRecommenderResponse(**response)

        return response
