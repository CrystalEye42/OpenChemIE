import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors


NODE_LINK_ATTRS = {
    "source": "from",
    "target": "to",
    "name": "id",
    "key": "key",
    "link": "edges",
}


def molecular_weight(smiles):
    """
    Calculate exact molecular weight for the given SMILES string

    Args:
        smiles: SMILES string for which to calculate molecular weight

    Returns:
         float: exact molecular weight
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        molwt = Descriptors.ExactMolWt(mol)

        return molwt
    except Exception:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        molwt = Descriptors.ExactMolWt(mol)

        return molwt


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


def json_to_nx_paths(paths):
    """
    Convert list of paths from json to networkx graphs.
    """
    if paths:
        if "nodes" in paths[0]:
            # Nodelink format
            try:
                paths = [nx.node_link_graph(path, attrs=NODE_LINK_ATTRS) for path in paths]
            except:
                paths = [nx.node_link_graph(path) for path in paths]

        else:
            # Treedata format
            paths = [tree_data_to_graph(path) for path in paths]
    return paths


def graph_to_json(graph, strip=False):
    """
    Convert graph to json in nodelink format.
    """
    graph_json = nx.node_link_data(graph)
    if strip:
        keys_to_keep = {"id", "smiles", "type"}

        graph_json["nodes"] = [
            {key: value for key, value in node.items() if key in keys_to_keep}
            for node in graph_json["nodes"]
        ]
    return graph_json


def augment_v2_graph(graph):
    """
    Add missing metadata from v2 graph.

    If ``extra=True``, computes additional metadata which is
    computed at runtime by the Python tree builder, but not by
    the C++ tree builder.

    Args:
        graph (nx.Graph): networkx graph to update metadata for
        extra (optional, bool): compute extra metadata
    """
    for node, node_data in graph.nodes(data=True):
        if node_data["type"] == "chemical":
            # Calculate molecular weight
            node_data["molwt"] = molecular_weight(node)

def augment_v2_uds(node_dict):
    """
    Add missing metadata from v2 graph.

    If ``extra=True``, computes additional metadata which is
    computed at runtime by the Python tree builder, but not by
    the C++ tree builder.

    Args:
        graph (nx.Graph): networkx graph to update metadata for
        extra (optional, bool): compute extra metadata
    """
    for node, node_data in node_dict.items():
        if node_data["type"] == "chemical":
            # Calculate molecular weight
            node_data["molwt"] = molecular_weight(node)


def calculate_path_metadata(paths, graph):
    """
    Calculate pathway-level metadata to support sorting.

    Args:
        paths (list): list of nx.Graph objects corresponding to pathways
        graph (nx.Graph): data graph containing full metadata for nodes
    """
    for path in paths:
        root_id = [v for v, d in path.in_degree() if d == 0][0]
        first_reaction_id = next(path.successors(root_id))
        first_reaction_smiles = path.nodes[first_reaction_id]["smiles"]

        try:
            path.graph["first_step_score"] = graph.nodes[first_reaction_smiles][
                "rxn_score_from_model"
            ]
        except KeyError:
            # for backward compatibility wit V1-imported results
            path.graph["first_step_score"] = graph.nodes[first_reaction_smiles][
                "template_score"
            ]

        path.graph["first_step_plausibility"] = graph.nodes[first_reaction_smiles].get(
            "plausibility", 0.0
        )

        all_reaction_ids = (
            n for n, d in path.nodes(data=True) if d["type"] == "reaction"
        )
        all_reaction_smiles = (path.nodes[n]["smiles"] for n in all_reaction_ids)
        all_reactions = [graph.nodes[smi] for smi in all_reaction_smiles]
        num_reactions = len(all_reactions)
        path.graph["num_reactions"] = num_reactions

        try:
            path.graph["avg_score"] = (
                sum(r["rxn_score_from_model"] for r in all_reactions) / num_reactions
            )
        except KeyError:
            # for backward compatibility wit V1-imported results
            path.graph["avg_score"] = (
                sum(r["template_score"] for r in all_reactions) / num_reactions
            )

        path.graph["avg_plausibility"] = (
            sum(r["plausibility"] for r in all_reactions) / num_reactions
        )

        try:
            path.graph["min_score"] = min(r["rxn_score_from_model"] for r in all_reactions)
        except KeyError:
            # for backward compatibility wit V1-imported results
            path.graph["min_score"] = min(r["template_score"] for r in all_reactions)

        path.graph["min_plausibility"] = min(r["plausibility"] for r in all_reactions)

        if "depth" not in path.graph:
            path.graph["depth"] = [
                path.nodes[v]["type"] for v in nx.dag_longest_path(path)
            ].count("reaction")

        precursor_ids = (v for v, d in path.out_degree() if d == 0)
        precursor_smiles = [path.nodes[n]["smiles"] for n in precursor_ids]
        if "precursor_cost" not in path.graph:
            precursor_prices = [
                graph.nodes[smi]["purchase_price"] for smi in precursor_smiles
            ]
            path.graph["precursor_cost"] = (
                sum(precursor_prices) if all(precursor_prices) else None
            )

        if "atom_economy" not in path.graph:
            target = path.nodes[root_id]["smiles"]
            target_molwt = graph.nodes[target]["molwt"]
            precursor_molwt = sum(graph.nodes[smi]["molwt"]
                                  for smi in precursor_smiles)
            path.graph["atom_economy"] = target_molwt / precursor_molwt

def uds2conn(uds: dict, format: str = ["graph", "unified_tree","pathways"]):
    node_dict = uds["node_dict"]
    uuid2smiles = uds["uuid2smiles"]
    connectivity = uds[format]
    
    if format == "graph":
        node_list = [prop for _, prop in node_dict.items()]
        data = {
            "directed": True,
            "multigraph": False,
            "nodes": node_list,
            "links": connectivity
        }
        graph = nx.node_link_graph(data)
        return graph
    elif format == "pathways":
        pathways = []
        for path_edges in connectivity:
            path_node_dict = {}
            for edge in path_edges:
                source_uuid = edge["source"]
                target_uuid = edge["target"]
                source_smiles = uuid2smiles[source_uuid]
                target_smiles = uuid2smiles[target_uuid]
                new_source_dict = {k:v for k, v in node_dict[source_smiles].items() if k != 'id'}
                new_target_dict = {k:v for k, v in node_dict[target_smiles].items() if k != 'id'}
                path_node_dict[source_uuid] = new_source_dict
                path_node_dict[target_uuid] = new_target_dict

            path_node_list = [{'id': node, **prop} for node, prop in path_node_dict.items()]
            data = {
                "directed": True,
                "multigraph": False,
                "nodes": path_node_list,
                "links": path_edges
            }
            path = nx.node_link_graph(data)
            pathways.append(path)
        return pathways


def get_graph_from_tree(tree: nx.DiGraph) -> nx.DiGraph:
    """Return cleaned version of original graph."""
    graph = tree.copy(as_view=False)
    # collapse tree into graph
    mapping = {}
    for node in graph:
        mapping[node] = node.split("_")[0]
    graph = nx.relabel_nodes(tree, mapping)
    return graph

def model_metadata_population(node_dict):
    models_predicted_by = []
    if node_dict.get("models_predicted_by"): # old tb result
        models_predicted_by = node_dict.get("models_predicted_by")
    elif node_dict.get("retro_backend") and node_dict.get("retro_model_name"): # old tb result
        backend = node_dict.get("retro_backend")
        model_name = node_dict.get("retro_model_name")
        model_score = node_dict.get("rxn_score_from_model")
        models_predicted_by = [[backend, model_name, model_score]]
    elif node_dict.get("model") and node_dict.get("trainingSet"): # old ipp result
        backend = node_dict.get("model")
        model_name = node_dict.get("trainingSet")
        model_score = node_dict.get("templateScore")
        models_predicted_by = [[backend, model_name, model_score]]

    model_metadata_list = []
    for model_info in models_predicted_by:
        backend, model_name, model_score = model_info
        
        template = {
            "count": node_dict.get("count"),
            "dimer_only": node_dict.get("dimer_only"),
            "index": node_dict.get("index"),
            "intra_only": node_dict.get("intra_only"),
            "necessary_reagent": node_dict.get("necessary_reagent") or node_dict.get("necessaryReagent"),
            "reaction_smarts": node_dict.get("reaction_smarts"),
            "template_set": model_name,
            "_id": None,
            "template_score": model_score,
            "template_rank": node_dict.get("templateRank"),
            "tforms": node_dict.get("tforms") or node_dict.get("templateIds"),
            "num_examples": node_dict.get("num_examples") or node_dict.get("numExamples")
        } 

        model_metadata = {
            "direction": "retro",
            "backend": backend,
            "model_name": model_name,
            "attributes": node_dict.get("attributes"), # old version does not have this
            "model_score": model_score,
            "normalized_model_score": model_score,
            "rank": node_dict.get("rank"),
            "reaction_id": node_dict.get("reaction_id"),
            "reaction_set": node_dict.get("reaction_set"),
        }

        if backend == "template_relevance":
            model_metadata.update({
                "source": {
                    "template": node_dict.get("template", template),
                    "reaction_data": None
                }
            })
        elif backend == "retrosim":
            model_metadata.update({
                "source": {
                    "template": {},
                    "reaction_data": node_dict.get("reaction_data")
                }
            })
        else:
            model_metadata.update({
                "source": {}
            })

        model_metadata_list.append(model_metadata)
    return model_metadata_list

def expand_one_conversion(node_dict):

    new_node_dict = {
        "smiles": node_dict.get("smiles") or node_dict.get("id"),
        "plausibility": node_dict["plausibility"],
        "rxn_score_from_model": node_dict["rxn_score_from_model"],
        "model_metadata": model_metadata_population(node_dict),
        "precursor_properties": {
            "rms_molwt": node_dict["rms_molwt"],
            "num_rings": node_dict["num_rings"],
            "scscore": node_dict["scscore"]
        },
        "precursor_rank": node_dict["rank"],
        "precursor_score": node_dict["rxn_score_from_model"],
        "reaction_properties": {
            "canonical_reaction_smiles": "CN(C)CCCl.OC(c1ccccc1)c1ccccc1>>CN(C)CCOC(c1ccccc1)c1ccccc1",
            "mapped_smiles": None,
            "plausibility": node_dict["plausibility"],
            "reacting_atoms": node_dict.get("reacting_atoms"),
            "selec_error": node_dict.get("selec_error"),
            "cluster_id": node_dict.get("group_id"),
            "cluster_name": node_dict.get("group_name")
        },
        "type": node_dict["type"],
        "id": node_dict.get("smiles") or node_dict.get("id"),
        "class_num": node_dict.get("class_num"),
        "class_name": node_dict.get("class_name")
    }
    return new_node_dict


def old2uds(json_paths, graph):
    from collections import defaultdict        
    pathways_connectivity = []
    pathways_properties = []
    uuid2smiles = {}
    for path in json_paths:
        nodes = path["nodes"]
        edges = path["links"]
        path_properties = path["graph"]
        
        clean_nodes = []
        for node in nodes:
            smiles = node["smiles"]
            id = node["id"]
            uuid2smiles[id] = smiles
            clean_nodes.append({"id": id})

        pathways_connectivity.append(edges)
        pathways_properties.append(path_properties)

    graph = nx.node_link_data(get_graph_from_tree(graph))
    graph_nodes = graph["nodes"] # graph connectivitiy id is smiles not uuid
    graph_connectivity = graph["links"] # graph connectivitiy id is smiles not uuid


    node_dict = defaultdict(dict)
    for node in graph_nodes:
        smiles = node.get("smiles") or node.get("id")
        assert smiles is not None, "SMILES"
        node["smiles"] = smiles

        if node_dict[smiles] and node_dict[smiles] != node:
            raise ValueError("Same smiles have different info.")
        if ">>" in smiles:
            node = expand_one_conversion(node)

        node_dict[smiles] = node

    uds = {
        "node_dict": node_dict,
        "uuid2smiles": uuid2smiles,
        "graph": graph_connectivity,
        "pathways": pathways_connectivity,
        "pathways_properties": pathways_properties,
    }

    return uds

def standardize_result_ipp(result: dict) -> dict:
    assert "dataGraph" in result.keys()
    for node in result["dataGraph"]["nodes"]:
        smiles = node.get("id")
        if ">>" in smiles:
            node["modelMetadata"] = model_metadata_population(node)
            node["PrecursorRank"] = node["precursorRank"]
            node["AvgModelScore"] = node["averageModelScore"]
    
    return result

def standardize_result_tb(result: dict) -> dict:
    """
    Takes a v2 tree builder result and convert to UDS format for frontend.
    """
    if "graph" in result.keys(): # old format to new UDS format
        json_data_graph = result["graph"]
        json_paths = result["paths"]

        if json_paths:
            paths = json_to_nx_paths(json_paths)
            json_paths = [graph_to_json(path) for path in paths] # standardize to nodelink data format

        graph = nx.node_link_graph(json_data_graph)

        result["uds"] = old2uds(json_paths, graph)
        del result["graph"]
        del result["paths"]

    graph = uds2conn(result["uds"], "graph")
    paths = uds2conn(result["uds"], "pathways")

    # Updates graph in place
    augment_v2_graph(graph)
    augment_v2_uds(result["uds"]["node_dict"]) # update uds node dictionary  

    # Calculate some pathway metadata
    calculate_path_metadata(paths, graph)
    for i, path in enumerate(paths):
        pathway_property = graph_to_json(path, strip=True).get("graph", {})
        result["uds"]["pathways_properties"][i].update(pathway_property)

    return result
