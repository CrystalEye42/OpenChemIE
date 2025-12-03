import re
import traceback
import uuid
from datetime import datetime
from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from schemas.cluster import ClusterSetting
from schemas.retro import RetroBackendOption
from typing import Annotated, Any, Literal
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry
from utils.tree_search_results import TreeSearchSavedResults
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class ExpandOneOptions(LowerCamelAliasModel):
    # aliasing to v1 fields
    template_max_count: int = Field(default=100, alias="template_count")
    template_max_cum_prob: int = Field(default=0.995, alias="max_cum_template_prob")
    banned_chemicals: list[str] = Field(
        default_factory=list,
        description="banned chemicals (in addition to user banned chemicals)",
        alias="forbidden_molecules",
        example=[]
    )
    banned_reactions: list[str] = Field(
        default_factory=list,
        description="banned reactions (in addition to user banned reactions)",
        alias="known_bad_reactions",
        example=[]
    )

    retro_backend_options: list[RetroBackendOption] = Field(
        default=[RetroBackendOption()],
        description="list of retro strategies to run in series"
    )
    use_fast_filter: bool = Field(
        default=True,
        description="whether to filter the results with the fast filter"
    )
    filter_threshold: float = Field(
        default=0.75,
        description="threshold for the fast filter"
    )
    retro_rerank_backend: Literal["relevance_heuristic", "scscore"] | None = Field(
        default=None,
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

    class Config:
        allow_population_by_field_name = True


class BuildTreeOptions(LowerCamelAliasModel):
    expansion_time: int = Field(
        default=30,
        description="max time for tree search in seconds",
        example=10
    )
    max_iterations: int | None = Field(
        default=None,
        description="max number of iterations"
    )
    max_chemicals: int | None = Field(
        default=None,
        description="max number of chemicals to explore"
    )
    max_reactions: int | None = Field(
        default=None,
        description="max number of reactions to explore"
    )
    max_templates: int | None = Field(
        default=None,
        description="max number of templates to explore"
    )
    max_branching: int = Field(
        default=25,
        description="max number of branching"
    )
    max_depth: int = Field(
        default=5,
        description="max tree depth"
    )
    exploration_weight: float = Field(
        default=1.0,
        description="weight for exploration (vs. exploitation)"
    )
    return_first: bool = Field(
        default=False,
        description="whether to stop when the first buyable path is found"
    )
    max_trees: int = Field(
        default=500,
        description="max number of buyable paths to explore"
    )

    # a bunch of termination logic. These were passed directly from the front end.
    # Grouping happens at the Django side. Let's tentatively keep that pattern for now.
    buyable_logic: Literal["none", "and", "or"] | None = Field(
        default="and",
        description="logic type for buyable termination"
    )
    max_ppg_logic: Literal["none", "and", "or"] | None = Field(
        default="none",
        description="logic type for price based termination"
    )
    max_ppg: float | None = Field(
        default=None,
        description="maximum price for price based termination"
    )
    max_scscore_logic: Literal["none", "and", "or"] | None = Field(
        default="none",
        description="logic type for synthetic complexity termination"
    )
    max_scscore: float | None = Field(
        default=None,
        description="maximum scscore for synthetic complexity termination"
    )
    chemical_property_logic: Literal["none", "and", "or"] | None = Field(
        default="none",
        description="logic type for chemical property termination"
    )
    max_chemprop_c: int | None = Field(
        default=None,
        description="maximum carbon count for termination"
    )
    max_chemprop_n: int | None = Field(
        default=None,
        description="maximum nitrogen count for termination"
    )
    max_chemprop_o: int | None = Field(
        default=None,
        description="maximum oxygen count for termination"
    )
    max_chemprop_h: int | None = Field(
        default=None,
        description="maximum hydrogen count for termination"
    )
    chemical_popularity_logic: Literal["none", "and", "or"] | None = Field(
        default="none",
        description="logic type for chemical popularity termination"
    )
    min_chempop_reactants: int | None = Field(
        default=5,
        description="minimum reactant precedents for termination"
    )
    min_chempop_products: int | None = Field(
        default=5,
        description="minimum product precedents for termination"
    )

    buyables_source: list[str] | None = Field(
        default=None,
        description="list of source(s) to consider when looking up buyables"
    )
    custom_buyables: list[str] | None = Field(
        default=None,
        description="list of chemicals to consider as buyable",
        example=[]
    )

    use_value_network: bool | None = Field(
        default=True,
        description="whether to use value_network for Vm computation"
    )

    class Config:
        @staticmethod
        def schema_extra(schema: dict[str, Any], model) -> None:
            # dropping the optional fields from the example
            del schema.get("properties")["max_iterations"]
            del schema.get("properties")["max_chemicals"]
            del schema.get("properties")["max_reactions"]
            del schema.get("properties")["max_templates"]
            del schema.get("properties")["buyables_source"]


class EnumeratePathsOptions(LowerCamelAliasModel):
    path_format: Literal["json", "graph"] = "json"
    json_format: Literal["treedata", "nodelink"] = "nodelink"
    sorting_metric: Literal[
        "plausibility",
        "number_of_starting_materials",
        "number_of_reactions",
        "score"
    ] = "plausibility"
    validate_paths: bool = True
    score_trees: bool = False
    cluster_trees: bool = False
    cluster_method: Literal["hdbscan", "kmeans"] = "hdbscan"
    min_samples: int = 5
    min_cluster_size: int = 5
    paths_only: bool = False
    max_paths: int = 200


class RetroStarInput(LowerCamelAliasModel):
    smiles: str = Field(
        description="target SMILES for Retro* tree building",
        example="CN(C)CCOC(c1ccccc1)c1ccccc1"
    )
    description: str | None = Field(
        default="",
        description="description of the Retro* task",
    )
    tags: str | None = Field(
        default="",
        description="tags of the Retro* task",
    )
    expand_one_options: ExpandOneOptions = Field(
        default_factory=ExpandOneOptions,
        description="options for one-step expansion"
    )
    build_tree_options: BuildTreeOptions = Field(
        default_factory=BuildTreeOptions,
        description="options for Retro* tree search"
    )
    enumerate_paths_options: EnumeratePathsOptions = Field(
        default_factory=EnumeratePathsOptions,
        description="options for path enumeration once the tree is built"
    )
    run_async: bool = False
    result_id: str = str(uuid.uuid4())


class UDS(BaseModel):
    node_dict: dict
    uuid2smiles: dict
    graph: list[dict]
    pathways: list[list[dict]]
    pathways_properties: list[dict]


class RetroStarResult(BaseModel):
    stats: dict[str, Any] | None
    uds: UDS | None
    version: int | str | None = "retro_star_0"
    result_id: str = ""


class RetroStarOutput(BaseModel):
    error: str
    status: str
    results: RetroStarResult


class RetroStarResponse(BaseResponse):
    result: RetroStarResult | None


@register_wrapper(
    name="tree_search_retro_star",
    input_class=RetroStarInput,
    output_class=RetroStarOutput,
    response_class=RetroStarResponse
)
class RetroStarWrapper(BaseWrapper):
    """Wrapper class for Retro* planning"""
    prefixes = ["tree_search/retro_star"]
    methods_to_bind: dict[str, list[str]] = {
        "get_config": ["GET"],
        "get_doc": ["GET"],
        "call_sync": ["POST"],
        "call_sync_without_token": ["POST"],
        "call_async": ["POST"],
        "retrieve": ["GET"]
    }

    def call_raw(self, input: RetroStarInput) -> RetroStarOutput:
        # Grouping for termination logics used to happen at the Django side.
        # Let's tentatively keep that pattern for now.
        dict_input = self.process_input(input)

        response = self.session_sync.post(
            self.prediction_url,
            json=dict_input,
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = RetroStarOutput(**output)

        return output

    def call_sync(
        self,
        input: RetroStarInput,
        token: Annotated[str, Depends(oauth2_scheme)]
    ) -> RetroStarResponse:
        """
        Endpoint for synchronous call to the Retro* planner.
        Login required for access to user banned lists.
        """
        # banned_chemicals handling, requires login token
        if not input.expand_one_options.banned_chemicals:
            input.expand_one_options.banned_chemicals = []
        banned_chemicals_controller = get_util_registry().get_util(
            module="banned_chemicals"
        )
        user_banned_chemicals = banned_chemicals_controller.get(token=token).__root__
        user_banned_chemicals = [entry.smiles for entry in user_banned_chemicals
                                 if entry.active]
        input.expand_one_options.banned_chemicals.extend(user_banned_chemicals)

        # banned_chemicals handling, requires login token
        if not input.expand_one_options.banned_reactions:
            input.expand_one_options.banned_reactions = []
        banned_reactions_controller = get_util_registry().get_util(
            module="banned_reactions"
        )
        user_banned_reactions = banned_reactions_controller.get(token=token).__root__
        user_banned_reactions = [entry.smiles for entry in user_banned_reactions
                                 if entry.active]
        input.expand_one_options.banned_reactions.extend(user_banned_reactions)

        results_controller = get_util_registry().get_util(
            module="tree_search_results_controller"
        )

        if input.run_async:
            results_controller.update_result_state(
                result_id=input.result_id,
                state="started",
                token=token
            )
        try:
            # actual backend call
            output = self.call_raw(input=input)
            response = self.convert_output_to_response(output)
            result_doc = response.result
            result_doc.result_id = input.result_id
        except Exception:
            if input.run_async:
                results_controller.update_result_state(
                    result_id=input.result_id,
                    state="failed",
                    token=token
                )
            raise HTTPException(
                status_code=500,
                detail=f"retro_star.call_sync() fails with the error: "
                       f"{traceback.format_exc()}"
            )

        if input.run_async:
            results_controller.update_result_state(
                result_id=input.result_id,
                state="completed",
                token=token
            )
            results_controller.save_results(
                result_id=input.result_id,
                result=result_doc,
                token=token
            )

        return response

    def call_sync_without_token(self, input: RetroStarInput) -> RetroStarResponse:
        """
        Endpoint for synchronous call to the Retro* planner.
        Skip login at the expense of losing access to user banned lists.
        """
        # banned_chemicals handling, does not require login token
        if not input.expand_one_options.banned_chemicals:
            input.expand_one_options.banned_chemicals = []

        # banned_chemicals handling, does not require login token
        if not input.expand_one_options.banned_reactions:
            input.expand_one_options.banned_reactions = []

        # actual backend call
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        print(input.expand_one_options)

        return response

    async def call_async(
        self,
        input: RetroStarInput,
        token: Annotated[str, Depends(oauth2_scheme)],
        priority: int = 0
    ) -> str:
        """
        Endpoint for asynchronous call to the Retro* planner.
        Login required for access to user banned lists.
        """
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        input.run_async = True
        input.result_id = str(uuid.uuid4())
        # Note that we can't use task_id as the result_id,
        # as it needs to be known beforehand
        settings = input.dict()
        settings = {k: v for k, v in settings.items() if "option" in k}

        saved_results = TreeSearchSavedResults(
            user=user.username,
            target_smiles=input.smiles,
            description=input.description,
            created=datetime.now(),
            modified=datetime.now(),
            result_id=input.result_id,
            result_state="pending",
            result_type="tree_builder",
            result=None,
            settings=settings,
            tags=[tag.strip() for tag in re.split(r"[,;]", input.tags)],
            shared_with=[user.username]
        )
        results_controller = get_util_registry().get_util(
            module="tree_search_results_controller"
        )
        results_controller.create(result=saved_results, token=token)

        from askcos2_celery.tasks import tree_search_retro_star_task
        async_result = tree_search_retro_star_task.apply_async(
            args=(self.name, input.dict(), token), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> RetroStarResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def process_input(input: RetroStarInput) -> dict:
        build_tree_options = input.build_tree_options
        dict_input = input.dict()

        termination_logic = {"and": [], "or": []}

        buyable_logic = build_tree_options.buyable_logic
        if buyable_logic != "none":
            termination_logic[buyable_logic].append("buyable")

        max_ppg_logic = build_tree_options.max_ppg_logic
        if max_ppg_logic != "none":
            max_ppg = build_tree_options.max_ppg
            termination_logic[max_ppg_logic].append("max_ppg")
        else:
            max_ppg = None

        max_scscore_logic = build_tree_options.max_scscore_logic
        if max_scscore_logic != "none":
            max_scscore = build_tree_options.max_scscore
            termination_logic[max_scscore_logic].append("max_scscore")
        else:
            max_scscore = None

        chemical_property_logic = build_tree_options.chemical_property_logic
        if chemical_property_logic != "none":
            param_dict = {
                "C": "max_chemprop_c",
                "N": "max_chemprop_n",
                "O": "max_chemprop_o",
                "H": "max_chemprop_h",
            }
            max_elements = {
                k: getattr(build_tree_options, v) for k, v in param_dict.items()
                if hasattr(build_tree_options, v)
            }
            termination_logic[chemical_property_logic].append("max_elements")
        else:
            max_elements = None

        chemical_popularity_logic = build_tree_options.chemical_popularity_logic
        if chemical_popularity_logic != "none":
            min_history = {
                "as_reactant": build_tree_options.min_chempop_reactants,
                "as_product": build_tree_options.min_chempop_products
            }
            termination_logic[chemical_popularity_logic].append("min_history")
        else:
            min_history = None

        dict_input["build_tree_options"].update({
            "max_ppg": max_ppg,
            "max_scscore": max_scscore,
            "max_elements": max_elements,
            "min_history": min_history,
            "termination_logic": termination_logic,
        })

        return dict_input

    @staticmethod
    def convert_output_to_response(output: RetroStarOutput
                                   ) -> RetroStarResponse:
        status_code = 200
        message = "retro_star.call_raw() successfully executed."
        result = output.results

        if not output.status == "SUCCESS":
            status_code = 500
            message = f"Backend error encountered during retro_star.call_raw() " \
                      f"with the following error message {output.error}"
            result = None

        response = RetroStarResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
