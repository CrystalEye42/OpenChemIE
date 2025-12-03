import copy
import numpy as np
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

    times = []
    successes = []

    for i, smi in enumerate(smiles):
        data = copy.deepcopy(query_template)
        data["smiles"] = smi.strip()
        resp = requests.post(
            url=f"http://{HOST}:{PORT}/api/tree-search/mcts/call-sync-without-token",
            json=data
        ).json()
        first_path_time = resp["result"]["stats"]["first_path_time"]
        total_paths = resp["result"]["stats"]["total_paths"]
        print(f"i: {i}, SMILES: {smi.strip()}, first_path_time: {first_path_time: .2f} s")

        success = total_paths > 0
        if success:
            times.append(first_path_time)
        successes.append(success)

    print(f"Average time for first path when found: {np.mean(times)}")
    print(f"Success rate: {np.mean(successes)}")


if __name__ == "__main__":
    main()
