from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ReactionClassificationInput(LowerCamelAliasModel):
    smiles: list[str] = Field(
        default="list of reaction SMILES to be classified; "
                "currently only supports prediction for a single SMILES",
        example=["CC(O)C.OC(=O)CC>>CC(OC(=O)CC)C"]
    )
    num_results: int = Field(
        default=10,
        description="number of top reaction classes to keep for each result"
    )


class ReactionClassificationResult(BaseModel):
    rank: float
    reaction_num: str
    reaction_name: str
    reaction_classnum: str
    reaction_classname: str
    reaction_superclassnum: str
    reaction_superclassname: str
    prediction_certainty: float


class ReactionClassificationOutput(BaseModel):
    error: str
    status: str
    results: list[ReactionClassificationResult]


class ReactionClassificationResponse(BaseResponse):
    result: list[ReactionClassificationResult] | None


@register_wrapper(
    name="reaction_classification",
    input_class=ReactionClassificationInput,
    output_class=ReactionClassificationOutput,
    response_class=ReactionClassificationResponse
)
class ReactionClassificationWrapper(BaseWrapper):
    """Wrapper class for Reaction Classification"""
    prefixes = ["reaction_classification"]

    def call_raw(self, input: ReactionClassificationInput
                 ) -> ReactionClassificationOutput:
        response = self.session_sync.post(
            f"{self.prediction_url}/reaction_class",
            json=input.dict(),
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = ReactionClassificationOutput(**output)

        return output

    def call_sync(self, input: ReactionClassificationInput
                  ) -> ReactionClassificationResponse:
        """
        Endpoint for synchronous call to the reaction classifier.
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: ReactionClassificationInput, priority: int = 0
                         ) -> str:
        """
        Endpoint for asynchronous call to the reaction classifier.
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> ReactionClassificationResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: ReactionClassificationOutput
                                   ) -> ReactionClassificationResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in reaction_classification " \
                      f"with the following error message {output.error}"
            result = None

        response = ReactionClassificationResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
