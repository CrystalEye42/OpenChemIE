import json
from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Literal
from wrappers import register_wrapper
from wrappers.base import BaseWrapper, BaseResponse


# load example from test case, as it is too long
try:
    example_file = "tests/wrappers/pathway_ranker/default_test_case_1.json"
    with open(example_file, "r") as f:
        example = json.load(f)
except FileNotFoundError:
    example = None


class PathwayRankerInput(LowerCamelAliasModel):
    trees: list[dict] = Field(
        description="list of trees to rank",
        example=[]
    )
    clustering: bool = Field(
        default=False,
        description="whether to cluster pathways"
    )
    cluster_method: Literal["hdbscan", "kmeans"] = Field(
        default="hdbscan",
        description="method to cluster pathways"
    )
    min_samples: int = Field(
        default=5,
        description="minimum samples for hdbscan"
    )
    min_cluster_size: int = Field(
        default=5,
        description="minimum cluster size for hdbscan"
    )

    if example is not None:
        class Config:
            schema_extra = {
                "example": example
            }


class PathwayRankerResult(BaseModel):
    scores: list
    encoded_trees: list
    clusters: list | None


class PathwayRankerOutput(BaseModel):
    error: str
    status: str
    results: list[PathwayRankerResult]


class PathwayRankerResponse(BaseResponse):
    result: PathwayRankerResult | None


@register_wrapper(
    name="pathway_ranker",
    input_class=PathwayRankerInput,
    output_class=PathwayRankerOutput,
    response_class=PathwayRankerResponse
)
class PathwayRankerWrapper(BaseWrapper):
    """Wrapper class for tree-LSTM based pathway ranker"""
    prefixes = ["pathway_ranker"]

    def call_sync(self, input: PathwayRankerInput) -> PathwayRankerResponse:
        """
        Endpoint for synchronous call to the tree-LSTM based pathway ranking service.
        https://pubs.rsc.org/en/content/articlehtml/2021/sc/d0sc05078d
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: PathwayRankerInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to the tree-LSTM based pathway ranking service.
        https://pubs.rsc.org/en/content/articlehtml/2021/sc/d0sc05078d
        """
        from askcos2_celery.tasks import pathway_ranker_task
        async_result = pathway_ranker_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> PathwayRankerResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: PathwayRankerOutput
                                   ) -> PathwayRankerResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results[0]
        else:
            status_code = 500
            message = f"Backend error encountered in pathway_ranker " \
                      f"with the following error message {output.error}"
            result = None

        response = PathwayRankerResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
