from adapters import register_adapter
from pydantic import BaseModel, Field
from typing import Literal
from wrappers.registry import get_wrapper_registry
from wrappers.cluster.default import (
    ClusterInput,
    ClusterResponse
)


class V1ClusterInput(BaseModel):
    original: str = Field(description="smiles string")
    outcomes: list[str] = Field(description="list of smiles strings of outcomes")
    feature: Literal["original", "outcomes", "all"] | None = Field(
        default="original",
        description="features to use"
    )
    fingerprint: str = Field(
        default="morgan",
        description="fingerprint type"
    )
    fpradius: int = Field(
        default=1,
        description="fingerprint radius"
    )
    fpnbits: int = Field(
        default=512,
        description="fingerprint bits"
    )
    clustermethod: Literal["hdbscan", "kmeans", "rxn_class"] | None = Field(
        default="kmeans",
        description="cluster method"
    )
    scores: list[float] | None = Field(
        default=None,
        description="list of scores of precursors"
    )


class V1ClusterAsyncReturn(BaseModel):
    request: V1ClusterInput
    task_id: str


class V1ClusterResult(BaseModel):
    output: list[int]
    names: dict[str, str]


@register_adapter(
    name="v1_cluster",
    input_class=V1ClusterInput,
    result_class=V1ClusterResult
)
class V1ClusterAdapter:
    """V1 Cluster Adapter"""
    prefix = "legacy/cluster"
    methods = ["POST"]

    def call_sync(self, input: V1ClusterInput) -> V1ClusterResult:
        wrapper = get_wrapper_registry().get_wrapper(module="cluster")
        wrapper_input = self.convert_input(input=input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)

        return response

    async def __call__(self, input: V1ClusterInput, priority: int = 0
                       ) -> V1ClusterAsyncReturn:
        from askcos2_celery.tasks import legacy_task

        async_result = legacy_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        async_return = V1ClusterAsyncReturn(
            request=input, task_id=task_id)

        return async_return

    # def __call__(self, input: V1ClusterInput) -> V1ClusterResult:
    #     return self.call_sync(input)

    @staticmethod
    def convert_input(input: V1ClusterInput) -> ClusterInput:
        wrapper_input = ClusterInput(
            original=input.original,
            outcomes=input.outcomes,
            feature=input.feature if input.feature is not None else "original",
            cluster_method=input.clustermethod if input.clustermethod is not None else "kmeans",
            fp_type=input.fingerprint if input.clustermethod is not None else "morgan",
            fp_length=input.fpnbits if input.fpnbits is not None else 512,
            fp_radius=input.fpradius if input.fpradius is not None else 1,
            scores=input.scores if input.scores is not None else None,
            classification_threshold=0.2,
        )

        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: ClusterResponse
                         ) -> V1ClusterResult:
        result = wrapper_response.result
        result = V1ClusterResult(output=result[0], names=result[1])

        return result
