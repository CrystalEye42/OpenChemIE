from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class SiteSelectivityInput(LowerCamelAliasModel):
    smiles: list[str] = Field(
        description="list of reactant SMILES for site selectivity prediction",
        example=["Cc1ccccc1"]
    )


class SiteSelectivityResult(BaseModel):
    smiles: str
    index: int
    task: str
    atom_scores: list[float]


class SiteSelectivityOutput(BaseModel):
    error: str
    status: str
    results: list[list[SiteSelectivityResult]]


class SiteSelectivityResponse(BaseResponse):
    result: list[list[SiteSelectivityResult]]


@register_wrapper(
    name="site_selectivity",
    input_class=SiteSelectivityInput,
    output_class=SiteSelectivityOutput,
    response_class=SiteSelectivityResponse
)
class SiteSelectivityWrapper(BaseWrapper):
    """Wrapper class for Site Selectivity"""
    prefixes = ["site_selectivity"]

    def call_sync(self, input: SiteSelectivityInput) -> SiteSelectivityResponse:
        """
        Endpoint for synchronous call to the site selectivity predictor.
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: SiteSelectivityInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to the site selectivity predictor.
        """
        from askcos2_celery.tasks import site_selectivity_task
        async_result = site_selectivity_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> SiteSelectivityResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: SiteSelectivityOutput
                                   ) -> SiteSelectivityResponse:

        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in site_selectivity " \
                      f"with the following error message {output.error}"
            result = None

        response = SiteSelectivityResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
