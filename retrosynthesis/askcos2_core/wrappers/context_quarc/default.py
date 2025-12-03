from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ContextQuarcInput(LowerCamelAliasModel):
    smiles: list[str] = Field(
        description="list of reaction SMILES",
        # Note this is just a single text-wrapped long SMILES
        example=[
            "CC(C)(C)OC(=O)O[C:1](=[O:2])[O:3][C:4]([CH3:5])([CH3:6])[CH3:7]."
            "[CH3:8][c:9]1[cH:10][c:11]([nH:12][cH:13]1)[CH:14]=[O:15]>"
            "CN(C)c1ccncc1.CC#N>"
            "[CH3:5][C:4]([CH3:6])([CH3:7])[O:3][C:1](=[O:2])[n:12]1[cH:13][c:9]([cH:10][c:11]1[CH:14]=[O:15])[CH3:8]"
        ]
    )
    top_k: int = Field(
        description="return top k results",
        example=3
    )


class ReactionAmount(BaseModel):
    reactant: str
    amount_range: str


class AgentAmount(BaseModel):
    agent: str
    amount_range: str


class ContextQuarcPredictions(BaseModel):
    rank: int
    agents: list[str]
    temperature: str
    reactant_amounts: list[ReactionAmount]
    agent_amounts: list[AgentAmount]
    score: float


class ContextQuarcResult(BaseModel):
    predictions: list[ContextQuarcPredictions]


class ContextQuarcOutput(BaseModel):
    status: str
    error: str
    results: list[ContextQuarcResult]


class ContextQuarcResponse(BaseResponse):
    result: list[ContextQuarcResult]


@register_wrapper(
    name="context_quarc",
    input_class=ContextQuarcInput,
    output_class=ContextQuarcOutput,
    response_class=ContextQuarcResponse
)
class ContextQuarcWrapper(BaseWrapper):
    """Wrapper class for Reaction Context Prediction with Quarc"""
    prefixes = ["context/quarc"]

    def call_raw(self, input: ContextQuarcInput) -> ContextQuarcOutput:
        input_as_dict = input.dict()

        response = self.session_sync.post(
            f"{self.prediction_url}",
            json=input_as_dict,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = ContextQuarcOutput(**output)

        return output

    def call_sync(self, input: ContextQuarcInput) -> ContextQuarcResponse:
        """
        Endpoint for synchronous call to quantitative reaction context
        recommendation using QUARC. https://doi.org/10.1039/D5SC04957A
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: ContextQuarcInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to quantitative reaction context
        recommendation using QUARC. https://doi.org/10.1039/D5SC04957A
        """
        from askcos2_celery.tasks import context_recommender_task
        async_result = context_recommender_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> ContextQuarcResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: ContextQuarcOutput
                                   ) -> ContextQuarcResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in context_quarc " \
                      f"with the following error message {output.error}"
            result = None

        response = ContextQuarcResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
