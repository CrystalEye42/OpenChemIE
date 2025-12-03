from adapters import register_adapter
from pydantic import BaseModel, Field
from typing import Literal
from wrappers.registry import get_wrapper_registry
from wrappers.forward.wldn5 import (
    ForwardWLDN5Input,
    ForwardWLDN5Response
)


class V1ForwardInput(BaseModel):
    reactants: str = Field(
        description="SMILES string of reactants"
    )
    reagents: str | None = Field(
        default=None,
        description="SMILES string of reagents"
    )
    solvent: str | None = Field(
        default=None,
        description="SMILES string of solvent"
    )
    training_set: str | None = Field(
        default="uspto_500k",
        description="model training set to use"
    )
    model_version: Literal[
        "wln_forward",
        "graph2smiles_forward_uspto_stereo",
        ""
    ] | None = Field(
        default="",
        description="model version to use for prediction"
    )
    num_results: int | None = Field(
        default=100,
        description="max number of results to return"
    )
    atommap: bool | None = Field(
        default=False,
        description="Flag to keep atom mapping from the prediction"
    )
    priority: int | None = Field(
        default=1,
        description="set priority for celery task (0 = low, "
                    "1 = normal (default), 2 = high)"
    )


class V1ForwardAsyncReturn(BaseModel):
    request: V1ForwardInput
    task_id: str


class V1ForwardOutput(BaseModel):
    rank: int
    score: float
    prob: float
    mol_wt: float
    smiles: str


class V1ForwardResult(BaseModel):
    __root__: list[V1ForwardOutput]


@register_adapter(
    name="v1_forward",
    input_class=V1ForwardInput,
    result_class=V1ForwardResult
)
class V1ForwardAdapter:
    """V1 Forward Adapter"""
    prefix = "legacy/forward"
    methods = ["POST"]

    def call_sync(self, input: V1ForwardInput) -> V1ForwardResult:
        wrapper = get_wrapper_registry().get_wrapper(module="forward_wldn5")
        wrapper_input = self.convert_input(input=input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)

        return response

    async def __call__(self, input: V1ForwardInput, priority: int = 0
                       ) -> V1ForwardAsyncReturn:
        from askcos2_celery.tasks import legacy_task

        async_result = legacy_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        async_return = V1ForwardAsyncReturn(
            request=input, task_id=task_id)

        return async_return

    # def __call__(self, input: V1ForwardInput) -> V1ForwardResult:
    #     return self.call_sync(input)

    @staticmethod
    def convert_input(input: V1ForwardInput) -> ForwardWLDN5Input:
        wrapper_input = ForwardWLDN5Input(
            model_name=input.training_set,
            reactants=input.reactants,
            contexts=None
        )

        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: ForwardWLDN5Response
                         ) -> V1ForwardResult:
        results = wrapper_response.result[0]
        converted_results = [V1ForwardOutput(
            rank=result.rank,
            score=result.score,
            prob=result.prob,
            mol_wt=result.mol_wt,
            smiles=result.outcome.smiles
        )for result in results]
        final_result = V1ForwardResult(__root__=converted_results)

        return final_result
