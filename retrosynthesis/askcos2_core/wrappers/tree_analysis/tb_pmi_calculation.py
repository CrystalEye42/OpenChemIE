import networkx as nx
import traceback
from wrappers.pmi_calculator.default import PmiCalculatorInput
from utils.tree_search_results_util import uds2conn
from wrappers.registry import get_wrapper_registry
from wrappers.tree_analysis.tree_analysis_utils import (
    clean_json,
    NIL_UUID,
    NODE_LINK_ATTRS
)


def _tb_pmi_calculation(tb_result: dict, index: int) -> tuple[dict, dict]:

    output = {
        "success": True,
        "error": None,
    }

    try:
        if index == -1:
            trees_nx = uds2conn(tb_result['uds'], "pathways")
        else:
            trees_nx = [uds2conn(tb_result['uds'], "pathways")[index]]
        json_paths = [
            clean_json(nx.tree_data(tree, NIL_UUID)) for tree in trees_nx
        ]
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        output["success"] = False
        output["error"] = f"Unable to load requested result." \
                        f"{traceback.format_exc()}"
        print("PMI calculation failed for tree builder result:", str(e))

        return tb_result, output

    try:
        # Run PMI calculator using JSON paths
        pmi_calculator = get_wrapper_registry().get_wrapper(module="pmi_calculator")
        wrapper_input = PmiCalculatorInput(trees=json_paths)
        res = pmi_calculator.call_sync(wrapper_input)
        pmis, message = res.result, res.message
        assert pmis, message
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        output["success"] = False
        output["error"] = f"PMI calculation failed." \
                          f"{traceback.format_exc()}"
        print("PMI calculation failed for tree builder result:", str(e))

        return tb_result, output


    try:
        pathways_properties = tb_result['uds']["pathways_properties"]
        if index == -1:
            for i, pathway_prop in enumerate(pathways_properties):
                pathway_prop["pmi"] = pmis[i]
        else:
            pathways_properties[index]["pmi"] = pmis[0]
            
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        output["success"] = False
        output["error"] = f"PMI calculation result processing failed." \
                          f"{traceback.format_exc()}"
        print("PMI calculation failed for tree builder result:", str(e))

        return tb_result, output

    return tb_result, output
