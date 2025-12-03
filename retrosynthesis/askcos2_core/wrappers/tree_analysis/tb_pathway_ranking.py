import networkx as nx
import traceback
from wrappers.pathway_ranker.default import PathwayRankerInput
from wrappers.registry import get_wrapper_registry
from utils.tree_search_results_util import uds2conn
from wrappers.tree_analysis.tree_analysis_utils import (
    clean_json,
    NIL_UUID,
    NODE_LINK_ATTRS
)



def _tb_pathway_ranking(
    tb_result: dict,
    clustering: bool,
    cluster_method: str,
    min_samples: int,
    min_cluster_size: int
) -> tuple[dict, dict]:
    """
    Run a pathway ranking prediction for a saved tree builder result.

    Args:
        tb_result (dict): tree builder saved result document

    Returns:
        dict, dict: result document and info dict with success and error fields
    """
    output = {
        "success": True,
        "error": None,
    }

    # TODO: Consolidate tree processing with duplicate code in results API
    
    try:
        trees_nx = uds2conn(tb_result['uds'], "pathways")
        json_paths = [
            clean_json(nx.tree_data(tree, NIL_UUID)) for tree in trees_nx
        ]
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        output["success"] = False
        output["error"] = f"Unable to load requested result. Traceback: " \
                          f"{traceback.format_exc()}"
        print("Pathway ranking failed for tree builder result:", str(e))

        return tb_result, output

    try:
        # Run pathway ranking and clustering using JSON paths
        pathway_ranker = get_wrapper_registry().get_wrapper(module="pathway_ranker")
        wrapper_input = PathwayRankerInput(
            trees=json_paths,
            clustering=clustering,
            cluster_method=cluster_method,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size
        )
        response = pathway_ranker.call_sync(wrapper_input)
        assert bool(response.result), response.message
        results = response.result.dict()
    except Exception as e:
        traceback.format_exc()
        output["success"] = False
        output["error"] = f"Pathway ranking prediction failed. Traceback: " \
                          f"{traceback.format_exc()}"

        return tb_result, output

    try:
        pathways_properties = tb_result['uds']["pathways_properties"]
        for i, pathway_prop in enumerate(pathways_properties):
            pathway_prop["score"] = results["scores"][i]
            pathway_prop["cluster_id"] = results["clusters"][i]
    except Exception as e:
        traceback.format_exc()
        output["success"] = False
        output["error"] = f"Pathway ranking prediction failed. Traceback: " \
                          f"{traceback.format_exc()}"

        return tb_result, output

    return tb_result, output
