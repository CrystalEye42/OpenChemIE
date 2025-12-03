import json
from adapters import register_adapter
from pydantic import BaseModel, Field
from typing import Optional, Literal
from wrappers.registry import get_wrapper_registry
from wrappers.pathway_ranker.default import (
    PathwayRankerInput,
    PathwayRankerResult,
    PathwayRankerResponse
)


class V1PathwayRankerInput(BaseModel):
    trees: str = Field(
        description="list of trees to rank as a json string"
    )
    cluster: bool | None = Field(
        default=True,
        description="whether to cluster pathways"
    )
    cluster_method: Literal["hdbscan", "kmeans"] | None = Field(
        default="hdbscan",
        description="cluster method for pathways"
    )
    min_samples: int | None = Field(
        default=5,
        description="min samples for hdbscan"
    )
    min_cluster_size: int | None = Field(
        default=5,
        description="min cluster size for hdbscan"
    )


class V1PathwayRankerAsyncReturn(BaseModel):
    request: V1PathwayRankerInput
    task_id: str


class V1PathwayRankerResult(BaseModel):
    __root__: PathwayRankerResult


@register_adapter(
    name="v1_pathway_ranker",
    input_class=V1PathwayRankerInput,
    result_class=V1PathwayRankerResult
)
class V1PathwayRankerAdapter:
    """V1 Pathway Ranker Adapter"""
    prefix = "legacy/pathway_ranker"
    methods = ["POST"]

    def call_sync(self, input: V1PathwayRankerInput) -> V1PathwayRankerResult:
        wrapper = get_wrapper_registry().get_wrapper(module="pathway_ranker")
        wrapper_input = self.convert_input(input=input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)

        return response

    async def __call__(self, input: V1PathwayRankerInput, priority: int = 0
                       ) -> V1PathwayRankerAsyncReturn:
        from askcos2_celery.tasks import legacy_task

        async_result = legacy_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        async_return = V1PathwayRankerAsyncReturn(
            request=input, task_id=task_id)

        return async_return

    # def __call__(self, input: V1PathwayRankerInput) -> V1PathwayRankerResult:
    #     return self.call_sync(input)

    @staticmethod
    def convert_input(input: V1PathwayRankerInput) -> PathwayRankerInput:
        trees_data = json.loads(input.trees)  # Convert JSON string to list of dictionaries
        wrapper_input = PathwayRankerInput(
            trees=trees_data,
            clustering=input.cluster,
            cluster_method=input.cluster_method,
            min_samples=input.min_samples,
            min_cluster_size=input.min_cluster_size
        )

        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: PathwayRankerResponse
                         ) -> V1PathwayRankerResult:
        print("wrapper_response is Here :::::::::::::::::::::::::::::::")
        print(wrapper_response)
        result = wrapper_response.result
        result = V1PathwayRankerResult(__root__=result)

        return result
