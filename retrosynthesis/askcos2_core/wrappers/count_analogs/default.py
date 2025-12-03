from pydantic import BaseModel
from schemas.base import LowerCamelAliasModel
from typing import Literal
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class CountAnalogsInput(LowerCamelAliasModel):
    reaction_smiles: list[str]
    reaction_smarts: list[str] = None
    atom_map_backend: Literal["rxnmapper", "indigo", "wln"] = "rxnmapper"
    min_plausibility: float = 0.1


class CountAnalogsOutput(BaseModel):
    error: str
    status: str
    results: int


class CountAnalogsResponse(BaseResponse):
    result: int


@register_wrapper(
    name="count_analogs",
    input_class=CountAnalogsInput,
    output_class=CountAnalogsOutput,
    response_class=CountAnalogsResponse
)
class CountAnalogsWrapper(BaseWrapper):
    """Wrapper class for Descriptors"""
    prefixes = ["count_analogs"]

    def call_sync(self, input: CountAnalogsInput) -> CountAnalogsResponse:
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: CountAnalogsInput, priority: int = 0) -> str:
        from askcos2_celery.tasks import count_analogs_task
        async_result = count_analogs_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> CountAnalogsResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: CountAnalogsOutput
                                   ) -> CountAnalogsResponse:
        response = {
            "status_code": 200,
            "message": "",
            "result": output.results
        }
        response = CountAnalogsResponse(**response)

        return response
