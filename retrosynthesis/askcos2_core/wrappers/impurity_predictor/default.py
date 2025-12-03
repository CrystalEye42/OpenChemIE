from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Literal
from wrappers import register_wrapper
from wrappers.base import BaseWrapper, BaseResponse


class ImpurityPredictorInput(LowerCamelAliasModel):
    predictor_backend: Literal[
        "augmented_transformer", "graph2smiles", "wldn5"] = Field(
        default="wldn5",
        description="backend for forward prediction"
    )
    predictor_model_name: str = Field(
        default="pistachio",
        description="backend model name for forward prediction"
    )
    inspector: str = Field(
        default="Reaxys inspector",
        description="reaction scorer to use"
    )
    atom_map_backend: Literal["rxnmapper", "indigo", "wln"] = Field(
        default="rxnmapper",
        description="backend for atom mapping"
    )
    topn_outcome: int = Field(
        default=3,
        description="max number of top results to return"
    )
    insp_threshold: float = Field(
        default=0.2,
        description="threshold for the inspector"
    )
    check_mapping: bool = Field(
        default=True,
        description="whether to check atom mapping"
    )

    rct_smi: str = Field(
        description="reactant SMILES",
        example="CN(C)CCCl.OC(c1ccccc1)c1ccccc1"
    )
    prd_smi: str = Field(
        default="",
        description="product SMILES"
    )
    sol_smi: str = Field(
        default="",
        description="solvent SMILES"
    )
    rea_smi: str = Field(
        default="",
        description="reagent SMILES"
    )


class ForwardResult(BaseModel):
    outcome: str
    score: float


class ImpurityResult(BaseModel):
    no: int
    prd_smiles: str
    prd_mw: float
    rct_rea_sol: list[dict]
    insp_scores: list[float]
    avg_insp_score: float
    similarity_to_major: float
    modes: list[int]
    modes_name: str


class ImpurityPredictorResult(BaseModel):
    predict_expand: list[ImpurityResult]
    predict_normal: list[ForwardResult]


class ImpurityPredictorOutput(BaseModel):
    error: str
    status: str
    results: ImpurityPredictorResult


class ImpurityPredictorResponse(BaseResponse):
    result: ImpurityPredictorResult | None


@register_wrapper(
    name="impurity_predictor",
    input_class=ImpurityPredictorInput,
    output_class=ImpurityPredictorOutput,
    response_class=ImpurityPredictorResponse
)
class ImpurityPredictorWrapper(BaseWrapper):
    """Wrapper class for Impurity Predictor"""
    prefixes = ["impurity_predictor"]

    def call_sync(self, input: ImpurityPredictorInput
                  ) -> ImpurityPredictorResponse:
        """
        Endpoint for synchronous call to the impurity prediction service.
        Primarily making use of the forward predictor to explore other reaction modes
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: ImpurityPredictorInput, priority: int = 0
                         ) -> str:
        """
        Endpoint for asynchronous call to the impurity prediction service.
        Primarily making use of the forward predictor to explore other reaction modes
        """
        from askcos2_celery.tasks import impurity_predictor_task
        async_result = impurity_predictor_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> ImpurityPredictorResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: ImpurityPredictorOutput
                                   ) -> ImpurityPredictorResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in impurity_predictor " \
                      f"with the following error message {output.error}"
            result = None

        response = ImpurityPredictorResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
