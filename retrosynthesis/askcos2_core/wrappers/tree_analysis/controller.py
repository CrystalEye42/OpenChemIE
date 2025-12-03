import json
from fastapi import Depends, HTTPException, Response
from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Annotated, Literal
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry
from wrappers import register_wrapper
from wrappers.base import BaseWrapper
from wrappers.tree_analysis.tb_count_analogs import _tb_count_analogs
from wrappers.tree_analysis.tb_pathway_ranking import _tb_pathway_ranking
from wrappers.tree_analysis.tb_pmi_calculation import _tb_pmi_calculation
from wrappers.tree_analysis.tb_reaction_classification \
    import _tb_reaction_classification


class TreeAnalysisInput(LowerCamelAliasModel):
    result_id: str = Field(description="tree result_id")
    task: Literal[
        "pathway_ranking",
        "reaction_classification",
        "pmi_calculation",
        "count_analogs",
    ] = Field(
        default="pathway_ranking",
        description="tree analysis task name"
    )

    # PMI calculator and count_analogs option
    index: int = Field(
        default=-1,
        description="only applicable for pmi_calculation and count_analogs. "
                    "Index of tree to be analyzed. "
                    "An index of -1 will analyze all trees."
    )

    # Pathway ranking options
    cluster_trees: bool = Field(
        default=True,
        description="only for pathway_ranking; whether to cluster trees"
    )
    cluster_method: Literal["hdbscan", "kmeans"] = Field(
        default="hdbscan",
        description="only for pathway_ranking; method to cluster trees"
    )
    cluster_min_samples: int = Field(
        default=5,
        description="minimum number of samples when using 'hdbscan'"
    )
    cluster_min_size: int = Field(
        default=5,
        description="minimum cluster size when using 'hdbscan'"
    )

    # Count analogs options
    min_plausibility: float = Field(
        default=0.1,
        description="only for count_analogs; minimum plausibility for fast filter"
    )
    atom_map_backend: Literal["rxnmapper", "indigo", "wln"] = Field(
        default="rxnmapper",
        description="only for count_analogs; atom mapping backend"
    )


class TreeAnalysisOutput(BaseModel):
    placeholder: str


class TreeAnalysisResponse(BaseModel):
    placeholder: str


@register_wrapper(
    name="tree_analysis_controller",
    input_class=TreeAnalysisInput,
    output_class=TreeAnalysisOutput,
    response_class=TreeAnalysisResponse
)
class TreeAnalysisController(BaseWrapper):
    """Tree Analysis Controller"""
    prefixes = ["tree_analysis/controller", "tree_analysis"]
    backend_wrapper_names = {
        "pathway_ranking": "pathway_ranker",
        "reaction_classification": "reaction_classification",
        "pmi_calculation": "pmi_calculator",
        "count_analogs": "count_analogs"
    }

    def __init__(self):
        pass        # TODO: proper inheritance

    def call_sync(
        self,
        input: TreeAnalysisInput,
        token: Annotated[str, Depends(oauth2_scheme)]
    ) -> Response:
        """
        Endpoint for synchronous call to the tree analysis controller,
        which dispatches the call to respective backend service
        """
        results_controller = get_util_registry().get_util(
            module="tree_search_results_controller"
        )
        result = results_controller.retrieve(result_id=input.result_id, token=token)
        result = result.dict()

        if result["result_type"] == "ipp":
            raise HTTPException(
                status_code=405,
                detail="Analysis of IPP results not supported."
            )

        result_id = input.result_id
        task = input.task

        output = {
            "result_id": result_id,
            "task": task,
            "success": True,
            "error": None,
        }

        try:
            tb_result = result["result"]
            new_revision = result.get("revision", 0) + 1
            if tb_result.get("version") != 2 and tb_result.get("version") != "retro_star_0":
                output["success"] = False
                output["error"] = \
                    "Unable to perform analysis tasks on legacy format " \
                    "tree builder results."

                return Response(
                    content=json.dumps(output),
                    status_code=500,
                    media_type="application/json"
                )
        except Exception as e:
            output["success"] = False
            output["error"] = "Unable to load requested result."
            print("Analysis task failed for tree builder result:", str(e))

            return Response(
                content=json.dumps(output),
                status_code=500,
                media_type="application/json"
            )

        if task == "pathway_ranking":
            result, info = _tb_pathway_ranking(
                tb_result=tb_result,
                clustering=input.cluster_trees,
                cluster_method=input.cluster_method,
                min_samples=input.cluster_min_samples,
                min_cluster_size=input.cluster_min_size
            )
        elif task == "reaction_classification":
            result, info = _tb_reaction_classification(tb_result=tb_result)
        elif task == "pmi_calculation":
            result, info = _tb_pmi_calculation(tb_result=tb_result, index=input.index)
        elif task == "count_analogs":
            result, info = _tb_count_analogs(
                tb_result=tb_result,
                index=input.index,
                min_plausibility=input.min_plausibility,
                atom_map_backend=input.atom_map_backend
            )
        else:
            output["success"] = False
            output["error"] = "Unrecognized tree analysis task type."

            return Response(
                content=json.dumps(output),
                status_code=405,
                media_type="application/json"
            )

        output.update(info)
        updated_result = {
            "result": result,
            "revision": new_revision
        }
        if info["success"]:
            resp = results_controller.update(
                result_id=input.result_id,
                updated_result=updated_result,
                token=token
            )
            if not resp.status_code == 200:
                return resp
            status_code = 200

        else:
            status_code = 500

        return Response(
            content=json.dumps(output),
            status_code=status_code,
            media_type="application/json"
        )

    async def call_async(
        self,
        input: TreeAnalysisInput,
        priority: int = 0,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> str:
        """
        Endpoint for asynchronous call to the tree analysis controller,
        which dispatches the call to respective backend service
        """
        from askcos2_celery.tasks import tree_analysis_task
        async_result = tree_analysis_task.apply_async(
            args=(self.name, input.dict(), token), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> Response:
        return await super().retrieve(task_id=task_id)
