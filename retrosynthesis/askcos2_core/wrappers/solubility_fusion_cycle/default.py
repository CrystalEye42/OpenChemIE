from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseWrapper


class SolubilityFusionCycleInput(LowerCamelAliasModel):
    solvent_smiles: list[str]
    solute_smiles: list[str]
    temperature: list[float]
    density: list | None = None

    class Config:
        schema_extra = {
            "example": {
                "solvent_smiles": ["CC(=O)O", "CC(=O)O"],
                "solute_smiles": ["C(CCC(=O)O)CC(=O)O", "C(CCC(=O)O)CC(=O)O"],
                "temperature": [298.0, 500.0],
                "density": [12.5, 0.0],
            }
        }


class SolubilityFusionCycleResult(BaseModel):
    solvent_smiles_canonical: str
    solute_smiles_canonical: str
    temperature: float = Field(alias="Temperature [K]")
    solvent_density: float
    SMILES: str
    MP_pred: float
    MP_std: float
    logS_calc: float
    gamma: float


class SolubilityFusionCycleOutput(BaseModel):
    error: str
    status: str
    results: list[SolubilityFusionCycleResult]


class ConvertedSolubilityResult(BaseModel):
    Solvent: str
    Solute: str
    Temp: float
    st_1: float
    log_st_1: float
    MP_pred: float
    MP_std: float
    gamma: float


class SolubilityFusionCycleResponse(BaseModel):
    __root__: list[ConvertedSolubilityResult] | None


@register_wrapper(
    name="solubility_fusion_cycle",
    input_class=SolubilityFusionCycleInput,
    output_class=SolubilityFusionCycleOutput,
    response_class=SolubilityFusionCycleResponse
)
class SolubilityFusionCycleWrapper(BaseWrapper):
    """Wrapper class for Fusion Cycle"""
    prefixes = ["solubility/fusion_cycle"]

    def call_sync(self, input: SolubilityFusionCycleInput) -> SolubilityFusionCycleResponse:
        """
        Endpoint for synchronous call to solubility prediction (fusion cycle) backend
        """
        if input.density is None:
            input.density = [0.0] * len(input.solute_smiles)
        else:
            for i, d in enumerate(input.density):
                if not d:
                    input.density[i] = 0.0
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: SolubilityFusionCycleInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to solubility prediction (fusion cycle) backend
        """
        if input.density is None:
            input.density = [0.0] * len(input.solute_smiles)
        else:
            for i, d in enumerate(input.density):
                if not d:
                    input.density[i] = 0.0

        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> SolubilityFusionCycleResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def _convert_fc_to_solubility(
        old_results: list[SolubilityFusionCycleResult]
    ) -> list[ConvertedSolubilityResult]:
        converted_results = []
        for r in old_results:
            converted_result = ConvertedSolubilityResult(
                Solvent=r.solvent_smiles_canonical,
                Solute=r.solute_smiles_canonical,
                Temp=r.temperature,
                st_1=ExactMolWt(
                    Chem.MolFromSmiles(r.solute_smiles_canonical)
                ) * 10.0 ** r.logS_calc,
                log_st_1=r.logS_calc,
                MP_pred=r.MP_pred,
                MP_std=r.MP_std,
                gamma=r.gamma
            )
            converted_results.append(converted_result)

        return converted_results

    def convert_output_to_response(self, output: SolubilityFusionCycleOutput
                                   ) -> SolubilityFusionCycleResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
            result = self._convert_fc_to_solubility(result)
        else:
            status_code = 500
            message = f"Backend error encountered in fusion cycle " \
                      f"with the following error message {output.error}"
            result = None

        response = SolubilityFusionCycleResponse(__root__=result)

        return response
