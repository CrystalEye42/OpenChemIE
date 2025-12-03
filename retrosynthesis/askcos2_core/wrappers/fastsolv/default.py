from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseWrapper


class FastSolvInput(LowerCamelAliasModel):
    solvent_smiles: list[str]
    solute_smiles: list[str]
    temperature: list[float]

    class Config:
        schema_extra = {
            "example": {
                "solvent_smiles": ["CC(=O)O", "CC(=O)O"],
                "solute_smiles": ["C(CCC(=O)O)CC(=O)O", "C(CCC(=O)O)CC(=O)O"],
                "temperature": [298.0, 500.0]
            }
        }


class FastSolvResult(BaseModel):
    solvent_smiles: str
    solute_smiles: str
    temperature: float
    predicted_logS: float
    predicted_logS_stdev: float


class FastSolvOutput(BaseModel):
    error: str
    status: str
    results: list[FastSolvResult]


class ConvertedSolubilityResult(BaseModel):
    Solvent: str
    Solute: str
    Temp: float
    st_1: float
    log_st_1: float
    uncertainty: float


class FastSolvResponse(BaseModel):
    # result: list[FastSolvResult]
    __root__: list[ConvertedSolubilityResult]


@register_wrapper(
    name="fastsolv",
    input_class=FastSolvInput,
    output_class=FastSolvOutput,
    response_class=FastSolvResponse
)
class FastSolvWrapper(BaseWrapper):
    """Wrapper class for FastSolv"""
    prefixes = ["fastsolv"]

    def call_sync(self, input: FastSolvInput) -> FastSolvResponse:
        """
        Endpoint for synchronous call to fastsolv prediction backend
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: FastSolvInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to fastsolv prediction backend
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> FastSolvResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def _convert_fastsolv_to_solubility(
        old_results: list[FastSolvResult]
    ) -> list[ConvertedSolubilityResult]:
        converted_results = []
        for r in old_results:
            converted_result = ConvertedSolubilityResult(
                Solvent=r.solvent_smiles,
                Solute=r.solute_smiles,
                Temp=r.temperature,
                st_1=ExactMolWt(
                    Chem.MolFromSmiles(r.solute_smiles)
                ) * 10.0 ** r.predicted_logS,
                log_st_1=r.predicted_logS,
                uncertainty=r.predicted_logS_stdev
            )
            converted_results.append(converted_result)

        return converted_results

    def convert_output_to_response(self, output: FastSolvOutput
                                   ) -> FastSolvResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
            result = self._convert_fastsolv_to_solubility(result)
        else:
            status_code = 500
            message = f"Backend error encountered in fastsolv " \
                      f"with the following error message {output.error}"
            result = None

        # response = FastSolvResponse(
        #     status_code=status_code,
        #     message=message,
        #     result=result
        # )
        response = FastSolvResponse(__root__=result)

        return response
