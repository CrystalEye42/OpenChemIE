from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper
from wrappers.context_quarc.default import (
    ContextQuarcInput,
    ContextQuarcOutput,
    ContextQuarcResult
)


class ContextQuarcSingleInput(LowerCamelAliasModel):
    reactants: str = Field(
        description="SMILES string of reactants",
        example="CC(C)(C)OC(=O)O[C:1](=[O:2])[O:3][C:4]([CH3:5])([CH3:6])[CH3:7]."
                "[CH3:8][c:9]1[cH:10][c:11]([nH:12][cH:13]1)[CH:14]=[O:15]"
    )
    products: str = Field(
        description="SMILES string of products",
        example="[CH3:5][C:4]([CH3:6])([CH3:7])[O:3][C:1](=[O:2])"
                "[n:12]1[cH:13][c:9]([cH:10][c:11]1[CH:14]=[O:15])[CH3:8]"
    )
    reagents: list[str] | None = Field(
        default=None,
        description="predefined reagents",
        example=["CN(C)c1ccncc1.CC#N"]
    )
    num_results: int = Field(
        default=10,
        description="max number of results to return",
        example=3
    )
    model: str = Field(
        default="graph",
        description="model backend; UNUSED"
    )


class ContextQuarcSingleOutput(BaseModel):
    status: str
    error: str
    results: ContextQuarcResult


class ContextQuarcSingleResponse(BaseResponse):
    result: ContextQuarcResult


@register_wrapper(
    name="context_quarc_single_query",
    input_class=ContextQuarcSingleInput,
    output_class=ContextQuarcSingleOutput,
    response_class=ContextQuarcSingleResponse
)
class ContextQuarcWrapper(BaseWrapper):
    """Wrapper class for Reaction Context Prediction with Quarc"""
    prefixes = ["context/quarc/single_query"]

    @staticmethod
    def convert_input(input: ContextQuarcSingleInput) -> ContextQuarcInput:
        if not input.reagents:
            input.reagents = None

        if input.reagents is None:
            smiles = [f"{input.reactants}>>{input.products}"]
        else:
            smiles = [
                f"{input.reactants}>{agent}>{input.products}"
                for agent in input.reagents
            ]

        new_input = ContextQuarcInput(
            smiles=smiles,
            top_k=input.num_results
        )

        return new_input

    @staticmethod
    def convert_output(output: ContextQuarcOutput) -> ContextQuarcSingleOutput:
        if output.status == "SUCCESS":
            results = output.results[0]
        else:
            results = None

        new_output = ContextQuarcSingleOutput(
            status=output.status,
            error=output.error,
            results=results
        )

        return new_output

    def call_raw(self, input: ContextQuarcInput) -> ContextQuarcOutput:
        input_as_dict = input.dict()

        response = self.session_sync.post(
            f"{self.prediction_url}",
            json=input_as_dict,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = ContextQuarcOutput(**output)

        return output

    def call_sync(self, input: ContextQuarcSingleInput) -> ContextQuarcSingleResponse:
        """
        Endpoint for synchronous call to quantitative reaction context
        recommendation using QUARC, single query. https://doi.org/10.1039/D5SC04957A
        """
        input = self.convert_input(input)
        output = self.call_raw(input=input)
        output = self.convert_output(output)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: ContextQuarcSingleInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to quantitative reaction context
        recommendation using QUARC, single query. https://doi.org/10.1039/D5SC04957A
        """
        from askcos2_celery.tasks import context_recommender_task
        async_result = context_recommender_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> ContextQuarcSingleResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: ContextQuarcSingleOutput
                                   ) -> ContextQuarcSingleResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in context_quarc " \
                      f"with the following error message {output.error}"
            result = None

        response = ContextQuarcSingleResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
