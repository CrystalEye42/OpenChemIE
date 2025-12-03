from adapters import register_adapter
from pydantic import BaseModel, Field
from wrappers.registry import get_wrapper_registry
from wrappers.context_recommender.default_uncleaned import (
    ContextRecommenderInput,
    ContextRecommenderResponse
)


def separate_names(smiles: str) -> tuple[str, str]:
    """
    Separate Reaxys names from SMILES
    """
    if "Reaxys" in smiles:
        parts = smiles.split(".")
        smiles = ".".join(part for part in parts if "Reaxys" not in part)
        names = ".".join(part for part in parts if "Reaxys" in part)
    else:
        names = ""

    return smiles, names


class V1ContextInput(BaseModel):
    reactants: str = Field(
        description="SMILES string of reactants"
    )
    products: str = Field(
        description="SMILES string of products"
    )
    with_smiles: bool = Field(
        default=False,
        description="whether to remove name-only predictions from results"
    )
    return_scores: bool = Field(
        default=False,
        description="whether to also return scores"
    )
    num_results: int = Field(
        default=10,
        description="max number of results to return"
    )


class V1ContextAsyncReturn(BaseModel):
    request: V1ContextInput
    task_id: str


class V1ContextResult(BaseModel):
    temperature: float = None
    solvent: str = ""
    reagent: str = ""
    catalyst: str = ""
    solvent_score: float = None
    best: bool
    solvent_name_only: str = ""
    reagent_name_only: str = ""
    catalyst_name_only: str = ""
    score: float = None


class V1ContextResults(BaseModel):
    __root__: list[V1ContextResult]


@register_adapter(
    name="v1_context",
    input_class=V1ContextInput,
    result_class=V1ContextResult
)
class V1ContextAdapter:
    """V1 Site Selectivity Adapter"""
    prefix = "legacy/context"
    methods = ["POST"]

    def call_sync(self, input: V1ContextInput) -> V1ContextResults:
        wrapper = get_wrapper_registry().get_wrapper(
            module="context_recommender_uncleaned"
        )
        wrapper_input = self.convert_input(input=input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(wrapper_response=wrapper_response)

        return response

    async def __call__(self, input: V1ContextInput, priority: int = 0
                       ) -> V1ContextAsyncReturn:
        from askcos2_celery.tasks import legacy_task

        async_result = legacy_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        async_return = V1ContextAsyncReturn(
            request=input, task_id=task_id)

        return async_return

    @staticmethod
    def convert_input(input: V1ContextInput) -> ContextRecommenderInput:
        wrapper_input = ContextRecommenderInput(
            smiles=f"{input.reactants}>>{input.products}",
            n_conditions=input.num_results,
            with_smiles=input.with_smiles,
            return_scores=input.return_scores
        )

        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: ContextRecommenderResponse
                         ) -> V1ContextResults:
        contexts = wrapper_response.result.conditions
        scores = wrapper_response.result.scores

        res = []
        for context in contexts:
            result = {
                "temperature": context[0],
                "solvent": context[1],
                "reagent": context[2],
                "catalyst": context[3],
                "solvent_score": context[4],
                "best": context[5]
            }

            result["solvent"], result["solvent_name_only"] = separate_names(
                result["solvent"]
            )
            result["reagent"], result["reagent_name_only"] = separate_names(
                result["reagent"]
            )
            result["catalyst"], result["catalyst_name_only"] = separate_names(
                result["catalyst"]
            )

            res.append(result)

        if scores is not None:
            for c, s in zip(res, scores):
                c["score"] = s

        result = V1ContextResults(__root__=res)

        return result
