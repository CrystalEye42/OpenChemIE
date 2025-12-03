from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from scipy.special import softmax
from typing import Literal
from utils.rdkit import molecular_weight
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper
from wrappers.forward.augmented_transformer import ForwardATInput, ForwardATResponse
from wrappers.forward.graph2smiles import ForwardG2SInput, ForwardG2SResponse
from wrappers.forward.wldn5 import ForwardWLDN5Input, ForwardWLDN5Response
from wrappers.registry import get_wrapper_registry


class ForwardInput(LowerCamelAliasModel):
    backend: Literal["augmented_transformer", "graph2smiles", "wldn5"] = Field(
        default="wldn5",
        description="backend for forward prediction",
    )

    model_name: str = Field(
        default="pistachio",
        description="model name for forward prediction backend"
    )
    smiles: list[str] = Field(
        description="list of reactant SMILES",
        example=["CS(=N)(=O)Cc1cccc(Br)c1.Nc1cc(Br)ccn1", "CCO.CC(=O)O"]
    )

    reagents: str = Field(
        default="",
        description="SMILES of reagents (dot-separated)"
    )
    solvent: str = Field(
        default="",
        description="SMILES of solvents (dot-separated)"
    )


class ForwardOutput(BaseModel):
    placeholder: str


class ForwardResult(BaseModel):
    rank: int
    outcome: str
    score: float
    prob: float
    mol_wt: float


class ForwardResponse(BaseResponse):
    result: list[list[ForwardResult]] | None


@register_wrapper(
    name="forward_controller",
    input_class=ForwardInput,
    output_class=ForwardOutput,
    response_class=ForwardResponse
)
class ForwardController(BaseWrapper):
    """Forward Controller"""
    prefixes = ["forward/controller", "forward"]
    backend_wrapper_names = {
        "augmented_transformer": "forward_augmented_transformer",
        "graph2smiles": "forward_graph2smiles",
        "wldn5": "forward_wldn5"
    }

    def __init__(self):
        pass        # TODO: proper inheritance

    def call_sync(self, input: ForwardInput) -> ForwardResponse:
        """
        Endpoint for synchronous call to the forward prediction controller,
        which dispatches the call to respective forward prediction backend service
        """
        module = self.backend_wrapper_names[input.backend]
        wrapper = get_wrapper_registry().get_wrapper(module=module)

        if input.reagents:
            for i, smi in enumerate(input.smiles):
                input.smiles[i] = smi + "." + input.reagents

        if input.solvent:
            for i, smi in enumerate(input.smiles):
                input.smiles[i] = smi + "." + input.solvent

        wrapper_input = self.convert_input(
            input=input, backend=input.backend)
        wrapper_response = wrapper.call_sync(wrapper_input)
        response = self.convert_response(
            wrapper_response=wrapper_response, backend=input.backend)

        return response

    async def call_async(self, input: ForwardInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to the forward prediction controller,
        which dispatches the call to respective forward prediction backend service
        """
        from askcos2_celery.tasks import forward_task
        async_result = forward_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> ForwardResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_input(
        input: ForwardInput, backend: str
    ) -> ForwardATInput | ForwardG2SInput | ForwardWLDN5Input:
        if backend == "augmented_transformer":
            wrapper_input = ForwardATInput(
                model_name=input.model_name,
                smiles=input.smiles
            )
        elif backend == "graph2smiles":
            wrapper_input = ForwardG2SInput(
                model_name=input.model_name,
                smiles=input.smiles
            )
        elif backend == "wldn5":
            wrapper_input = ForwardWLDN5Input(
                model_name=input.model_name,
                reactants=input.smiles,
                contexts=None
            )
        else:
            raise ValueError(f"Unsupported forward backend: {backend}!")

        return wrapper_input

    @staticmethod
    def convert_response(
        wrapper_response:
            ForwardATResponse | ForwardG2SResponse | ForwardWLDN5Response,
        backend: str
    ) -> ForwardResponse:
        status_code = wrapper_response.status_code
        message = wrapper_response.message

        if not status_code == 200:
            response = {
                "status_code": status_code,
                "message": message,
                "result": None
            }
            response = ForwardResponse(**response)

            return response

        if backend in ["augmented_transformer", "graph2smiles"]:
            # list[dict] -> list[list[dict]]
            result = []
            for result_per_smi in wrapper_response.result:
                normalized_scores = softmax(result_per_smi.scores)
                result.append(
                    [{
                        "rank": i + 1,
                        "outcome": outcome,
                        "score": score,
                        "prob": float(normalized_score),
                        "mol_wt": molecular_weight(outcome)
                    } for i, (outcome, score, normalized_score) in enumerate(zip(
                        result_per_smi.products,
                        result_per_smi.scores,
                        normalized_scores
                    ))]
                )
        elif backend == "wldn5":
            # list[list[dict]] -> list[list[dict]]
            result = []
            for result_per_smi in wrapper_response.result:
                result.append(
                    [{
                        "rank": i + 1,
                        "outcome": raw_result.outcome.smiles,
                        "score": raw_result.score,
                        "prob": raw_result.prob,
                        "mol_wt": molecular_weight(raw_result.outcome.smiles)
                    } for i, raw_result in enumerate(result_per_smi)]
                )
        else:
            raise ValueError(f"Unsupported forward backend: {backend}!")

        response = {
            "status_code": status_code,
            "message": message,
            "result": result
        }
        response = ForwardResponse(**response)

        return response
