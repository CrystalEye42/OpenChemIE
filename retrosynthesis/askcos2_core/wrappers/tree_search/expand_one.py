from fastapi import Depends
from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from schemas.cluster import ClusterSetting
from schemas.retro import RetroBackendOption
from typing import Annotated, Any, Literal
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ExpandOneInput(LowerCamelAliasModel):
    smiles: str = Field(
        description="target SMILES for one-step expansion",
        example="CN(C)CCOC(c1ccccc1)c1ccccc1"
    )
    retro_backend_options: list[RetroBackendOption] = Field(
        default=[RetroBackendOption()],
        description="list of retro strategies to run in series"
    )
    banned_chemicals: list[str] = Field(
        default_factory=list,
        description="banned chemicals (in addition to user banned chemicals)",
        example=[]
    )
    banned_reactions: list[str] = Field(
        default_factory=list,
        description="banned reactions (in addition to user banned reactions)",
        example=[]
    )
    use_fast_filter: bool = Field(
        default=True,
        description="whether to filter the results with the fast filter"
    )
    fast_filter_threshold: float = Field(
        default=0.75,
        description="threshold for the fast filter"
    )
    retro_rerank_backend: Literal["relevance_heuristic", "scscore"] | None = Field(
        default="relevance_heuristic",
        description="backend for retro rerank"
    )
    atom_map_backend: Literal["indigo", "rxnmapper", "wln"] = Field(
        default="rxnmapper",
        description="backend for atom map"
    )
    cluster_precursors: bool = Field(
        default=False,
        description="whether to cluster proposed precursors"
    )
    cluster_setting: ClusterSetting = Field(
        default_factory=ClusterSetting,
        description="settings for clustering"
    )
    extract_template: bool = Field(
        default=False,
        description="whether to extract templates "
                    "(mostly for template-free suggestions)"
    )
    return_reacting_atoms: bool = Field(
        default=True,
        description="whether to return the indices of reacting atoms"
    )
    selectivity_check: bool = Field(
        default=False,
        description="whether to perform quick selectivity check "
                    "by reverse application of the forward template"
    )

class ModelMetadata(BaseModel):
    direction: str
    backend: str
    model_name: str
    attributes: dict
    model_score: float
    normalized_model_score: float
    rank: int
    reaction_id: str | None
    reaction_set: str | None
    source: dict

class ModelReactionProperties(BaseModel):
    canonical_reaction_smiles: str
    mapped_smiles: str
    plausibility: float
    reacting_atoms: list[int] | None
    selec_error: bool | None
    cluster_id: int | None
    cluster_name: str | None

class RetroResult(BaseModel):
    # from retro_controller
    average_model_score: float
    model_metadata: list[ModelMetadata]
    outcome: str
    precursor_properties: dict
    precursor_rank: int
    precursor_score: float
    reaction_properties: ModelReactionProperties


class ExpandOneOutput(BaseModel):
    error: str
    status: str
    results: list[RetroResult]


class ExpandOneResponse(BaseResponse):
    result: list[RetroResult] | dict[int, list[RetroResult]] | None


@register_wrapper(
    name="tree_search_expand_one",
    input_class=ExpandOneInput,
    output_class=ExpandOneOutput,
    response_class=ExpandOneResponse
)
class ExpandOneWrapper(BaseWrapper):
    """Wrapper class for ones tep expansion"""
    prefixes = ["tree_search/expand_one"]
    methods_to_bind: dict[str, list[str]] = {
        "get_config": ["GET"],
        "get_doc": ["GET"],
        "call_sync": ["POST"],
        "call_sync_without_token": ["POST"],
        "call_async": ["POST"],
        "retrieve": ["GET"]
    }

    def call_raw(self, input: ExpandOneInput) -> ExpandOneOutput:
        cache_controller = get_util_registry().get_util(module="cache_controller")
        try:
            response = cache_controller.get(module_name=self.name, input=input)
            if isinstance(response, dict):
                response = ExpandOneOutput(**response)
        except KeyError:
            response = self.session_sync.post(
                self.prediction_url,
                json=input.dict(),
                timeout=self.config["deployment"]["timeout"]
            )
            response = response.json()
            response = ExpandOneOutput(**response)

            # only add successful response into cache
            if response.status == "SUCCESS":
                cache_controller.add(
                    module_name=self.name,
                    input=input,
                    response=response
                )

        return response

    def call_sync(
        self,
        input: ExpandOneInput,
        token: Annotated[str, Depends(oauth2_scheme)]
    ) -> ExpandOneResponse:
        """
        Endpoint for synchronous call to the one-step expansion engine,
        which combines retro prediction, retro rerank and postprocessing.
        Login required for access to user banned lists.
        """
        # banned_chemicals handling, requires login token
        if not input.banned_chemicals:
            input.banned_chemicals = []

        try:
            banned_chemicals_controller = get_util_registry().get_util(
                module="banned_chemicals"
            )
            user_banned_chemicals = banned_chemicals_controller.get(token=token).__root__
            user_banned_chemicals = [entry.smiles for entry in user_banned_chemicals
                                     if entry.active]
            input.banned_chemicals.extend(user_banned_chemicals)
        except:
            pass

        # banned_chemicals handling, requires login token
        if not input.banned_reactions:
            input.banned_reactions = []

        try:
            banned_reactions_controller = get_util_registry().get_util(
                module="banned_reactions"
            )
            user_banned_reactions = banned_reactions_controller.get(token=token).__root__
            user_banned_reactions = [entry.smiles for entry in user_banned_reactions
                                     if entry.active]
            input.banned_reactions.extend(user_banned_reactions)
        except:
            pass

        # actual backend call
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output, input)

        return response

    def call_sync_without_token(self, input: ExpandOneInput) -> ExpandOneResponse:
        """
        Endpoint for synchronous call to the one-step expansion engine,
        which combines retro prediction, retro rerank and postprocessing.
        Skip login at the expense of losing access to user banned lists.
        """
        # banned_chemicals handling, does not require login token
        if not input.banned_chemicals:
            input.banned_chemicals = []

        # banned_chemicals handling, does not require login token
        if not input.banned_reactions:
            input.banned_reactions = []

        # actual backend call
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output, input)

        return response

    async def call_async(
        self,
        input: ExpandOneInput,
        token: Annotated[str, Depends(oauth2_scheme)],
        priority: int = 0
    ) -> str:
        """
        Endpoint for asynchronous call to the one-step expansion engine,
        which combines retro prediction, retro rerank and postprocessing.
        Login required for access to user banned lists.
        """
        from askcos2_celery.tasks import tree_search_expand_one_task
        async_result = tree_search_expand_one_task.apply_async(
            args=(self.name, input.dict(), token), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> ExpandOneResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(
        output: ExpandOneOutput,
        input: ExpandOneInput
    ) -> ExpandOneResponse:
        status_code = 200
        message = "expand_one.call_raw() successfully executed."
        result = output.results

        if not output.status == "SUCCESS":
            status_code = 500
            message = f"Backend error encountered during expand_one.call_raw() " \
                      f"with the following error message {output.error}"
            result = None

        response = ExpandOneResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
