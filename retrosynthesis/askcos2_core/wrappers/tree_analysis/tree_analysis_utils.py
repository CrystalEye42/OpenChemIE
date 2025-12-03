import networkx as nx

NIL_UUID = "00000000-0000-0000-0000-000000000000"
NODE_LINK_ATTRS = {
    "source": "from",
    "target": "to",
    "name": "id",
    "key": "key",
    "link": "edges",
}

OUTPUT_KEYS = {
    "chemical": [
        "smiles",
        "id",
        "as_reactant",
        "as_product",
        "ppg",
        "properties",
        "terminal",
    ],
    "reaction": [
        "smiles",
        "id",
        "plausibility",
        "forward_score",
        "model_metadata",
        "precursor_properties",
        "precursor_rank",
        "precursor_score",
        "reaction_properties"
    ],
}

# Map from keys used by tree builder graph to name used in pathways
PATH_KEY_DICT = {
    "smiles": "smiles",
    "type": "type",
    "id": "id",
    "as_reactant": "as_reactant",
    "as_product": "as_product",
    "plausibility": "plausibility",
    "forward_score": "forward_score",   # From graph optimization
    "ppg": "ppg",               # If calling clean_json on a previously cleaned tree
    "purchase_price": "ppg",
    "rxn_score_from_model": "rxn_score_from_model",
    "terminal": "terminal",
    "model_metadata": "model_metadata",
    "precursor_properties": "precursor_properties",
    "precursor_rank": "precursor_rank",
    "precursor_score": "precursor_score",
    "reaction_properties": "reaction_properties"
}



def clean_json(path):
    """
    Clean up json representation of a pathway. Accepts paths from either
    tree builder version.

    Note about chemical/reaction node identification:
        * For treedata format, chemical nodes have an ``is_chemical`` attribute,
          while reaction nodes have an ``is_reaction`` attribute
        * For nodelink format, all nodes have a ``type`` attribute, whose value
          is either ``chemical`` or ``reaction``

    This distinction in the JSON schema is for historical reasons and is not
    an official aspect of the treedata/nodelink formats.
    """

    def _add_missing_fields(_node, _type):
        for _key in OUTPUT_KEYS[_type]:
            if _key not in _node:
                _node[_key] = None

    if "nodes" in path:
        # Node link format
        nodes = []
        for node in path["nodes"]:
            new_node = {
                PATH_KEY_DICT[key]: value
                for key, value in node.items()
                if key in PATH_KEY_DICT
            }
            # Set any output keys not present to None
            _add_missing_fields(new_node, node["type"])
            nodes.append(new_node)
        path["nodes"] = nodes
        output = path
    else:
        # Tree data format
        output = {}
        for key, value in path.items():
            if key == "type":
                if value == "chemical":
                    output["is_chemical"] = True
                elif value == "reaction":
                    output["is_reaction"] = True
            elif key == "children":
                output["children"] = [clean_json(c) for c in value]
            elif key in PATH_KEY_DICT:
                output[PATH_KEY_DICT[key]] = value

        # Set any output keys not present to None
        _add_missing_fields(output, path["type"])

        if "children" not in output:
            output["children"] = []

    return output


def nx_paths_to_json(paths, root_uuid, json_format="nodelink"):
    """
    Convert list of paths from networkx graphs to json.
    """
    if json_format == "treedata":
        # Include graph attributes at top level of resulting json
        return [
            {"attributes": path.graph, **clean_json(nx.tree_data(path, root_uuid))}
            for path in paths
        ]
    elif json_format == "nodelink":
        return [
            clean_json(nx.node_link_data(path, attrs=NODE_LINK_ATTRS)) for path in paths
        ]
    else:
        raise ValueError(f"Unsupported value for json_format: {json_format}")


def tree_data_to_graph(tree):
    """
    Convert a single tree from tree data format to networkx graph.

    Args:
        tree (dict): tree in networkx tree data json format

    Returns:
        tree (nx.DiGraph): tree as networkx graph
    """
    # Extract attributes if they exist
    attributes = tree.pop("attributes", {})

    # Create graph from json
    tree = nx.tree_graph(tree)

    # Assign attributes to graph
    tree.graph.update(attributes)

    for node, node_data in tree.nodes.items():
        if node_data.get("is_chemical"):
            node_data["type"] = "chemical"
            del node_data["is_chemical"]
            if "terminal" not in node_data:
                # Set terminal attribute based on number of successors
                node_data["terminal"] = tree.out_degree(node) == 0
        elif node_data.get("is_reaction"):
            node_data["type"] = "reaction"
            del node_data["is_reaction"]

    return tree
