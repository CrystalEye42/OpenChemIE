from adapters import register_adapter
from pydantic import BaseModel, Field
from typing import Literal
from wrappers.registry import get_wrapper_registry
from wrappers.context_recommender.predict_graph import (
    ContextRecommenderGraphInput,
    ContextRecommenderGraphResponse
)
from wrappers.context_recommender.predict_fp import (
    ContextRecommenderFPInput,
    ContextRecommenderFPResponse
)


class V1ContextV2Input(BaseModel):
    reactants: str = Field(
        description="SMILES string of reactants"
    )
    products: str = Field(
        description="SMILES string of products"
    )
    reagents: list[str] | None = Field(
        default=None,
        description="predefined reagents"
    )
    num_results: int = Field(
        default=10,
        description="max number of results to return. "
                    "It is also the beam width in beam search."
    )
    model: Literal["graph", "fp"] = Field(
        default="graph",
        description="model backend"
    )


class V1ContextV2AsyncReturn(BaseModel):
    request: V1ContextV2Input
    task_id: str


class V1ContextV2Result(BaseModel):
    reagents_score: float
    temperature: float
    reagents: dict[str, float]
    reactants: dict[str, float]


class V1ContextV2Results(BaseModel):
    __root__: list[V1ContextV2Result]


@register_adapter(
    name="v1_context_v2",
    input_class=V1ContextV2Input,
    result_class=V1ContextV2Results
)
class V1ContextV2Adapter:
    """V1 Context-v2 Adapter"""
    prefix = "legacy/context_v2"
    methods = ["POST"]
    backend_wrapper_names = {
        "graph": "context_recommender_pr_graph",
        "fp": "context_recommender_pr_fp"
    }

    def call_sync(self, input: V1ContextV2Input) -> V1ContextV2Results:
        module = self.backend_wrapper_names[input.model]
        wrapper = get_wrapper_registry().get_wrapper(module=module)
        wrapper_input = self.convert_input(input=input, backend=input.model)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)

        return response

    async def __call__(self, input: V1ContextV2Input, priority: int = 0
                       ) -> V1ContextV2AsyncReturn:
        from askcos2_celery.tasks import legacy_task

        async_result = legacy_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        async_return = V1ContextV2AsyncReturn(
            request=input, task_id=task_id)

        return async_return

    @staticmethod
    def convert_input(input: V1ContextV2Input, backend: str
                      ) -> ContextRecommenderFPInput | ContextRecommenderGraphInput:
        if backend == "graph":
            wrapper_input_class = ContextRecommenderGraphInput
        elif backend == "fp":
            wrapper_input_class = ContextRecommenderFPInput
        else:
            raise ValueError(f"Unsupported context v2 backend: {backend}!")

        wrapper_input = wrapper_input_class(
            smiles=f"{input.reactants}>>{input.products}",
            reagents=input.reagents,
            n_conditions=input.num_results
        )

        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: ContextRecommenderFPResponse |
                         ContextRecommenderGraphResponse) -> V1ContextV2Results:
        result = V1ContextV2Results(__root__=wrapper_response.result)

        return result
