from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class QMDescriptorsInput(LowerCamelAliasModel):
    smiles: list[str] = Field(
        description="list of reactant SMILES for QM Descriptors prediction",
        example=["Cc1ccccc1"]
    )


class QMDescriptorsResult(BaseModel):
    smiles: str
    npa_e: list[float] = Field(alias="npa charge (e)")
    npa_e_pos: list[float] = Field(alias="npa charge + (e)")
    npa_e_neg: list[float] = Field(alias="npa charge - (e)")
    npa_parr_func_e_pos: list[float] = Field(alias="npa parr function + (e)")
    npa_parr_func_e_neg: list[float] = Field(alias="npa parr function - (e)")
    sheilding_constant_ppm: list[float] = Field(alias="shielding constant (ppm)")
    valence_1s: list[float] = Field(alias="1s valence orbital occupancy (e)")
    valence_2s: list[float] = Field(alias="2s valence orbital occupancy (e)")
    valence_2p: list[float] = Field(alias="2p valence orbital occupancy (e)")
    valence_3s: list[float] = Field(alias="3s valence orbital occupancy (e)")
    valence_3p: list[float] = Field(alias="3p valence orbital occupancy (e)")
    valence_4s: list[float] = Field(alias="4s valence orbital occupancy (e)")
    valence_4p: list[float] = Field(alias="4p valence orbital occupancy (e)")
    bond_index: list[float] = Field(alias="bond index (unitless)")
    bond_length: list[float] = Field(alias="bond length (Å)")
    bond_charge: list[float] = Field(alias="bond charge (e)")
    natural_ion: list[float] = Field(alias="natural ionicity (unitless)")
    dipole_moment: float = Field(alias="dipole moment (debye)")
    traceless: float = Field(alias="traceless quadrupole moment (debye⋅Å)")
    HOMO_3_LUMO: float = Field(alias="HOMO-3/LUMO (hartree)")
    HOMO_3_LUMO_1: float = Field(alias="HOMO-3/LUMO+1 (hartree)")
    HOMO_3_LUMO_2: float = Field(alias="HOMO-3/LUMO+2 (hartree)")
    HOMO_3_LUMO_3: float = Field(alias="HOMO-3/LUMO+3 (hartree)")
    HOMO_2_LUMO: float = Field(alias="HOMO-2/LUMO (hartree)")
    HOMO_2_LUMO_1: float = Field(alias="HOMO-2/LUMO+1 (hartree)")
    HOMO_2_LUMO_2: float = Field(alias="HOMO-2/LUMO+2 (hartree)")
    HOMO_2_LUMO_3: float = Field(alias="HOMO-2/LUMO+3 (hartree)")
    HOMO_1_LUMO: float = Field(alias="HOMO-1/LUMO (hartree)")
    HOMO_1_LUMO_1: float = Field(alias="HOMO-1/LUMO+1 (hartree)")
    HOMO_1_LUMO_2: float = Field(alias="HOMO-1/LUMO+2 (hartree)")
    HOMO_1_LUMO_3: float = Field(alias="HOMO-1/LUMO+3 (hartree)")
    HOMO_LUMO: float = Field(alias="HOMO/LUMO (hartree)")
    HOMO_LUMO_1: float = Field(alias="HOMO/LUMO+1 (hartree)")
    HOMO_LUMO_2: float = Field(alias="HOMO/LUMO+2 (hartree)")
    HOMO_LUMO_3: float = Field(alias="HOMO/LUMO+3 (hartree)")
    IP: float = Field(alias="IP (hartree)")
    EA: float = Field(alias="EA (hartree)")


class QMDescriptorsOutput(BaseModel):
    error: str
    status: str
    results: list[QMDescriptorsResult]


class QMDescriptorsResponse(BaseResponse):
    result: list[QMDescriptorsResult]


@register_wrapper(
    name="qm_descriptors",
    input_class=QMDescriptorsInput,
    output_class=QMDescriptorsOutput,
    response_class=QMDescriptorsResponse
)
class QMDescriptorsWrapper(BaseWrapper):
    """Wrapper class for QM Descriptors"""
    prefixes = ["qm_descriptors"]

    def call_sync(self, input: QMDescriptorsInput) -> QMDescriptorsResponse:
        """
        Endpoint for synchronous call to the QM Descriptors predictor.
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response
    
    async def call_async(self, input: QMDescriptorsInput, priority: int = 0
                         ) -> str:
        """
        Endpoint for asynchronous call to the qm descriptors.
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> QMDescriptorsResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: QMDescriptorsOutput
                                   ) -> QMDescriptorsResponse:

        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in qm_descriptors " \
                      f"with the following error message {output.error}"
            result = None

        response = QMDescriptorsResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
