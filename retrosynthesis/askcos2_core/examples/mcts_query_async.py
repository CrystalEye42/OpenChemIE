import copy
import os
import requests


query_template = {
    "smiles": "",
    "expand_one_options": {
        "template_max_count": 100,
        "template_max_cum_prob": 0.995,
        "banned_chemicals": [],
        "banned_reactions": [],
        "retro_backend_options": [
            {
                "retro_backend": "template_relevance",
                "retro_model_name": "reaxys",
                "max_num_templates": 100,
                "max_cum_prob": 0.995,
                "attribute_filter": []
            }
        ],
        "use_fast_filter": True,
        "filter_threshold": 0.75,
        "cluster_precursors": False,
        "cluster_setting": {
            "feature": "original",
            "cluster_method": "hdbscan",
            "fp_type": "morgan",
            "fp_length": 512,
            "fp_radius": 1,
            "classification_threshold": 0.2
        },
        "extract_template": False,
        "return_reacting_atoms": True,
        "selectivity_check": False
    },
    "build_tree_options": {
        "expansion_time": 60,
        "max_branching": 25,
        "max_depth": 6,
        "max_iterations": 500,
        "exploration_weight": 1,
        "return_first": True
    },
    "enumerate_paths_options": {
        "path_format": "json",
        "json_format": "nodelink",
        "sorting_metric": "plausibility",
        "validate_paths": True,
        "score_trees": False,
        "cluster_trees": False,
        "cluster_method": "hdbscan",
        "min_samples": 5,
        "min_cluster_size": 5,
        "paths_only": False,
        "max_paths": 200
    },
}


def main():
    HOST = "0.0.0.0"
    PORT = "9100"

    with open("examples/chembl_100.txt", "r") as f:
        smiles = f.readlines()

    s = requests.Session()
    # The resp object itself is not used. The access token will be stored in
    # the session cookie after the call below is made.
    resp = s.post(
        url=f"http://{HOST}:{PORT}/api/admin/token",
        data={
            "username": os.environ.get("ASKCOS_USERNAME"),
            "password": os.environ.get("ASKCOS_PASSWORD"),
        }
    )

    for i, smi in enumerate(smiles[:2]):
        data = copy.deepcopy(query_template)
        data["smiles"] = smi.strip()
        data["description"] = f"Test_1, {i}, {smi.strip()}"

        # The access token will be automatically sent in the cookie with the request.
        # There is no more need for sending the authentication header.
        # Note: must use s.post (NOT requests.post) to send the correct session cookie
        resp = s.post(
            url=f"http://{HOST}:{PORT}/api/tree-search/mcts/call-async",
            json=data
        )

        print(f"i: {i}, SMILES: {smi.strip()}, result_id: {resp.text}")


if __name__ == "__main__":
    main()
