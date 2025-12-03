import copy
from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Literal
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ForwardWLDN5Input(LowerCamelAliasModel):
    model_name: Literal["pistachio", "uspto_500k"] = Field(
        default="pistachio",
        description="model name for WLDN5 backend"
    )
    reactants: str | list[str] = Field(
        description="list of reactant SMILES",
        example=["CS(=N)(=O)Cc1cccc(Br)c1.Nc1cc(Br)ccn1", "CCO.CC(=O)O"]
    )
    contexts: list[str] | None = Field(
        default=None,
        description="not used; only for legacy compatibility",
        example=[]
    )


class ForwardWLDN5Outcome(BaseModel):
    smiles: str
    template_ids: list = None
    num_examples: int = 0


class ForwardWLDN5Result(BaseModel):
    rank: int
    outcome: ForwardWLDN5Outcome
    score: float
    prob: float
    mol_wt: float


class ForwardWLDN5Output(BaseModel):
    status: str
    error: str
    results: list[list[ForwardWLDN5Result]]


class ForwardWLDN5Response(BaseResponse):
    result: list[list[ForwardWLDN5Result]] | None


@register_wrapper(
    name="forward_wldn5",
    input_class=ForwardWLDN5Input,
    output_class=ForwardWLDN5Output,
    response_class=ForwardWLDN5Response
)
class ForwardWLDN5Wrapper(BaseWrapper):
    """Wrapper class for Forward Prediction with WLDN5"""
    prefixes = ["forward/wldn5"]

    def call_raw(self, input: ForwardWLDN5Input) -> ForwardWLDN5Output:
        if isinstance(input.reactants, str):
            input.reactants = [input.reactants]

        input_as_dict = copy.deepcopy(input.dict())
        results = []
        for i, reactants in enumerate(input.reactants):
            input_as_dict["reactants"] = reactants
            response = self.session_sync.post(
                f"{self.prediction_url}",
                json=input_as_dict,
                timeout=self.config["deployment"]["timeout"]
            )
            result = response.json()["results"]
            results.extend(result)

        output = response.json()            # hardcoding using the last response status
        output["results"] = results
        output = ForwardWLDN5Output(**output)

        return output

    def call_sync(self, input: ForwardWLDN5Input) -> ForwardWLDN5Response:
        """
        Endpoint for synchronous call to WLDN5 forward predictor.
        https://pubs.rsc.org/en/content/articlehtml/2019/sc/c8sc04228d
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: ForwardWLDN5Input, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to WLDN5 forward predictor.
        https://pubs.rsc.org/en/content/articlehtml/2019/sc/c8sc04228d
        """
        from askcos2_celery.tasks import forward_task
        async_result = forward_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> ForwardWLDN5Response | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: ForwardWLDN5Output
                                   ) -> ForwardWLDN5Response:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in forward/wldn5 " \
                      f"with the following error message {output.error}"
            result = None

        response = ForwardWLDN5Response(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
