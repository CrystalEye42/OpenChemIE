import networkx as nx
import traceback as tb
from wrappers.count_analogs.default import CountAnalogsInput
from wrappers.tree_analysis.tree_analysis_utils import NODE_LINK_ATTRS
from wrappers.registry import get_wrapper_registry
from utils.registry import get_util_registry
from utils.tree_search_results_util import uds2conn
from wrappers.tree_analysis.tree_analysis_utils import (
    clean_json,
    NIL_UUID,
    NODE_LINK_ATTRS
)


def _tb_count_analogs_combinations(
    tree: nx.Graph,
    min_plausibility: float,
    atom_map_backend: str = "rxnmapper"
) -> int:
    """
    Run Count Combinations for calculating analogs

    Returns:
        int: result of count combinations
    """
    # FIXME together with the backend.. pricer.lookup_smarts not working
    graph_enumerator = get_wrapper_registry().get_wrapper(module="count_analogs")
    template_controller = get_util_registry().get_util(module="template")
    reaction_smiles = [
        d["smiles"] for v, d in tree.nodes(data=True) if d["type"] == "reaction"
    ]
    reaction_smarts = None
    """
    try:
        template_ids = [
            d["tforms"][0] for v, d in tree.nodes(data=True)
            if d["type"] == "reaction"
        ]
    except:
        template_ids = [
            d["id"] for v, d in tree.nodes(data=True)
            if d["type"] == "reaction" and "id" in d
        ]

    reaction_smarts = [
        template_controller.find_one_by_id(_id=t)["reaction_smarts"]
        for t in template_ids
    ]
    """
    wrapper_input = CountAnalogsInput(
        reaction_smiles=reaction_smiles,
        reaction_smarts=reaction_smarts,
        min_plausibility=min_plausibility,
        atom_map_backend=atom_map_backend
    )
    count = graph_enumerator.call_sync(wrapper_input).result

    return count


def _tb_count_analogs(
    tb_result: dict,
    index: int,
    min_plausibility: float,
    atom_map_backend: str = "rxnmapper"
) -> tuple[dict, dict]:
    """
    Run a reaction classification prediction for a saved tree builder result.

    Returns:
        dict, dict: result document and info dict with success and error fields
    """


    output = {
        "success": True,
        "error": None,
    }
    try:
        if index == -1:
            trees_nx = uds2conn(tb_result['uds'], "pathways")
        else:
            trees_nx = [uds2conn(tb_result['uds'], "pathways")[index]]
        
        # count analogs only support nodelink format
        graph_paths = trees_nx

    except Exception as e:
        output["success"] = False
        output["error"] = f"Unable to load requested result." \
                        f"{tb.format_exc()}"
        print("Counting analogs failed for tree builder result:", str(e))

        return tb_result, output

    try:
        num_analogs = []
        print(f"Counting analogs for {len(graph_paths)} trees")
        for i, tree in enumerate(graph_paths):
            num_analogs.append(
                _tb_count_analogs_combinations(
                    tree=tree,
                    min_plausibility=min_plausibility,
                    atom_map_backend=atom_map_backend
                )
            )
    except Exception as e:
        output["success"] = False
        output["error"] = f"Analog counting failed." \
                          f"{tb.format_exc()}"
        tb.print_exc()

        return tb_result, output

    try:
        pathways_properties = tb_result["uds"]["pathways_properties"]
        if index == -1:
            for i, pathway_prop in enumerate(pathways_properties):
                pathway_prop["num_analogs"] = num_analogs[i]
        else:
            pathways_properties[index]["num_analogs"] = num_analogs[0]
            
    except Exception as e:
        output["success"] = False
        output["error"] = f"Analog counting result processing failed." \
                        f"{tb.format_exc()}"
        tb.print_exc()

        return tb_result, output

    return tb_result, output
