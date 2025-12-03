import json
from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseWrapper, BaseResponse


# load example from test case, as it is too long
try:
    example_file = "tests/wrappers/pathway_ranker/default_test_case_1.json"
    with open(example_file, "r") as f:
        example = json.load(f)
except FileNotFoundError:
    example = None


class PmiCalculatorInput(LowerCamelAliasModel):
    trees: list[dict] = Field(
        description="list of trees to calculate PMI",
        example=example.get("trees", []) if example is not None else []
    )


class PmiCalculatorOutput(BaseModel):
    error: str
    status: str
    results: list[list[float]]


class PmiCalculatorResponse(BaseResponse):
    result: list[float] | None


@register_wrapper(
    name="pmi_calculator",
    input_class=PmiCalculatorInput,
    output_class=PmiCalculatorOutput,
    response_class=PmiCalculatorResponse
)
class PmiCalculatorWrapper(BaseWrapper):
    """Wrapper class for Pmi Calculator"""
    prefixes = ["pmi_calculator"]

    def call_sync(self, input: PmiCalculatorInput) -> PmiCalculatorResponse:
        """
        Endpoint for synchronous call to the process mass intensity (PMI) calculator.
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)
        return response

    async def call_async(self, input: PmiCalculatorInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to the process mass intensity (PMI) calculator.
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> PmiCalculatorResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: PmiCalculatorOutput
                                   ) -> PmiCalculatorResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results[0]
        else:
            status_code = 500
            message = f"Backend error encountered in pmi_calculator " \
                      f"with the following error message {output.error}"
            result = None

        response = PmiCalculatorResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
