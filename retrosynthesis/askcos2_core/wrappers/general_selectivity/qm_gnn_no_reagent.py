from pydantic import BaseModel
from schemas.base import LowerCamelAliasModel
from typing import Literal
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class GeneralSelectivityInput(LowerCamelAliasModel):
    smiles: str
    atom_map_backend: Literal["indigo", "rxnmapper", "wln"] = "wln"
    mapped: bool = False
    all_outcomes: bool = False
    no_map_reagents: bool = False


class GeneralSelectivityResult(BaseModel):
    smiles: str
    prob: float
    rank: int


class GeneralSelectivityOutput(BaseModel):
    status: str
    error: str
    results: list[GeneralSelectivityResult]


class GeneralSelectivityResponse(BaseResponse):
    result: list[GeneralSelectivityResult] | None


@register_wrapper(
    name="general_selectivity_qm_gnn_no_reagent",
    input_class=GeneralSelectivityInput,
    output_class=GeneralSelectivityOutput,
    response_class=GeneralSelectivityResponse
)
class GeneralSelectivityWrapper(BaseWrapper):
    """Wrapper class for general selectivity, QM GNN model, no reagent"""
    prefixes = ["general_selectivity/qm_gnn_no_reagent"]

    def call_raw(self, input: GeneralSelectivityInput) -> GeneralSelectivityOutput:
        response = self.session_sync.post(
            f"{self.prediction_url}/qm_no_reagent",
            json=input.dict(),
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = GeneralSelectivityOutput(**output)

        return output

    def call_sync(self, input: GeneralSelectivityInput) -> GeneralSelectivityResponse:
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: GeneralSelectivityInput, priority: int = 0
                         ) -> str:
        from askcos2_celery.tasks import general_selectivity_task
        async_result = general_selectivity_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> GeneralSelectivityResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: GeneralSelectivityOutput
                                   ) -> GeneralSelectivityResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in " \
                      f"general_selectivity/qm_gnn_no_reagent " \
                      f"with the following error message {output.error}"
            result = None

        response = GeneralSelectivityResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
