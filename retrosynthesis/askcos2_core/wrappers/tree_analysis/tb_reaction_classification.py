import networkx as nx
import traceback
from collections import defaultdict
from wrappers.reaction_classification.get_top_class_batch import GetTopClassBatchInput
from wrappers.registry import get_wrapper_registry
from utils.tree_search_results_util import uds2conn
from wrappers.tree_analysis.tree_analysis_utils import (
    NIL_UUID,
    NODE_LINK_ATTRS,
    nx_paths_to_json,
    tree_data_to_graph
)


def _tb_reaction_classification(tb_result: dict) -> tuple[dict, dict]:
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
        graph = uds2conn(tb_result['uds'], "graph")
        reactions = [v for v, d in graph.nodes(data=True) if d["type"] == "reaction"]
        reaction_classifier = get_wrapper_registry().get_wrapper(
            module="get_top_class_batch"
        )
        wrapper_input = GetTopClassBatchInput(smiles=reactions)
        res = reaction_classifier.call_sync(wrapper_input)
        rxn_classes, message = res.result, res.message
        assert rxn_classes, message
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        output["success"] = False
        output["error"] = f"Reaction classification prediction failed." \
                        f"{traceback.format_exc()}"
        print("Reaction classification failed for tree builder result:", str(e))

        return tb_result, output

    try:
        for i, rxn in enumerate(reactions):
            rxn_data = graph.nodes[rxn]
            rxn_data["class_num"], rxn_data["class_name"] = rxn_classes[i]
        
        graph_json = nx.node_link_data(graph)
        graph_nodes = graph_json["nodes"] # graph connectivitiy id is smiles not uuid
        graph_connectivity = graph_json["links"] # graph connectivitiy id is smiles not uuid

        node_dict = defaultdict(dict)
        for node in graph_nodes:
            smiles = node["smiles"]
            if node_dict[smiles] and node_dict[smiles] != node:
                raise ValueError("Same smiles have different info.")
            node_dict[smiles] = node

        tb_result["uds"]["node_dict"] = node_dict
        tb_result["uds"]["graph_connectivity"] = graph_connectivity
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        output["success"] = False
        output["error"] = f"Reaction classification result processing failed." \
                        f"{traceback.format_exc()}"
        print("Reaction classification failed for tree builder result:", str(e))
        return tb_result, output

    return tb_result, output
