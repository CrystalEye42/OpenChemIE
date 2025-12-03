from pydantic import BaseModel
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class GetTopClassBatchInput(LowerCamelAliasModel):
    smiles: list[str]
    level: int = 2
    threshold: float = 0.2


class GetTopClassBatchOutput(BaseModel):
    error: str
    status: str
    results: list[tuple[str, str]]


class GetTopClassBatchResponse(BaseResponse):
    result: list[tuple[str, str]]


@register_wrapper(
    name="get_top_class_batch",
    input_class=GetTopClassBatchInput,
    output_class=GetTopClassBatchOutput,
    response_class=GetTopClassBatchResponse
)
class GetTopClassBatchWrapper(BaseWrapper):
    """Wrapper class for getting top reaction classes for a batch of SMILES"""
    prefixes = [
        "get_top_class_batch",
        "reaction_classification/get_top_class_batch"
    ]

    def call_raw(self, input: GetTopClassBatchInput
                 ) -> GetTopClassBatchOutput:
        minibatch_size = 5
        minibatch_results = []

        for i in range(0, len(input.smiles), minibatch_size):
            minibatch_input = {
                "smiles": input.smiles[i:i+minibatch_size],
                "level": input.level,
                "threshold": input.threshold
            }
            response = self.session_sync.post(
                f"{self.prediction_url}/get_top_class_batch",
                json=minibatch_input,
                timeout=self.config["deployment"]["timeout"]
            )
            minibatch_result = response.json()["results"]
            error = response.json()["error"]
            status = response.json()["status"]

            if error:
                break
            minibatch_results.extend(minibatch_result)

        output = {
            "status": status,
            "error": error,
            "results": minibatch_results
        }
        output = GetTopClassBatchOutput(**output)

        return output

    def call_sync(self, input: GetTopClassBatchInput
                  ) -> GetTopClassBatchResponse:
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: GetTopClassBatchInput, priority: int = 0
                         ) -> str:
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> GetTopClassBatchResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: GetTopClassBatchOutput
                                   ) -> GetTopClassBatchResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in get_top_class_batch " \
                      f"with the following error message {output.error}"
            result = None

        response = GetTopClassBatchResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
