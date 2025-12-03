from adapters import register_adapter
from pydantic import BaseModel, Field
from wrappers.registry import get_wrapper_registry
from wrappers.solubility.default import (
    SolubilityInput,
    SolubilityResult,
    SolubilityOutput,
    SolubilityResponse
)


class V1SolubilityInput(BaseModel):
    solvent: str = Field(
        description="solvent SMILES"
    )
    solute: str = Field(
        description="solute SMILES"
    )
    temp: float = Field(
        description="temperature for solubility prediction (K)"
    )
    ref_solvent: str | None = Field(
        default=None,
        description="reference solvent SMILES"
    )
    ref_solubility: float | None = Field(
        default=None,
        description="solute solubility in reference solvent as logS (log10(mol/L))"
    )
    ref_temp: float | None = Field(
        default=None,
        description="temperature for reference solubility (K)"
    )
    hsub298: float | None = Field(
        default=None,
        description="sublimation enthalpy of solute at 298 K (kcal/mol)"
    )
    cp_gas_298: float | None = Field(
        default=None,
        description="gas phase heat capacity of solute at 298 K (cal/K/mol)"
    )
    cp_solid_298: float | None = Field(
        default=None,
        description="solid phase heat capacity of solute at 298 K (cal/K/mol)"
    )


class V1SolubilityLegacyInput(BaseModel):
    task_list: list[V1SolubilityInput]


class V1SolubilityAsyncReturn(BaseModel):
    request: V1SolubilityInput
    task_id: str


class V1SolubilityResult(BaseModel):
    __root__: list[SolubilityResult]


@register_adapter(
    name="v1_solubility",
    input_class=V1SolubilityInput,
    result_class=V1SolubilityResult
)
class V1SolubilityAdapter:
    """V1 Solubility Adapter"""
    prefix = "legacy/solubility"
    methods = ["POST"]

    def call_sync(self, input: V1SolubilityInput) -> V1SolubilityResult:
        wrapper = get_wrapper_registry().get_wrapper(module="solubility")
        wrapper_input = self.convert_input(input=input)
        print(wrapper_input)
        wrapper_response = wrapper.call_sync(wrapper_input)
        print("Here is the out put")
        print(wrapper_response)
        response = self.convert_response(wrapper_response=wrapper_response)

        return response

    async def __call__(self, input: V1SolubilityInput, priority: int = 0
                       ) -> V1SolubilityAsyncReturn:
        from askcos2_celery.tasks import legacy_task

        async_result = legacy_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        async_return = V1SolubilityAsyncReturn(
            request=input, task_id=task_id)

        return async_return

    @staticmethod
    def convert_input(input: V1SolubilityInput) -> SolubilityInput:
        wrapper_input = SolubilityInput(task_list=[input])
        return wrapper_input

    @staticmethod
    def convert_response(wrapper_response: SolubilityResponse
                         ) -> V1SolubilityResult:
        result = wrapper_response

        return result
