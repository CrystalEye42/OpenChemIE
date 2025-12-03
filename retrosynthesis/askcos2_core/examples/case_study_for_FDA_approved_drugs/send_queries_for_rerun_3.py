import copy
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
                "max_num_templates": 1000,
                "max_cum_prob": 0.999,
                "attribute_filter": []
            },
            {
                "retro_backend": "augmented_transformer",
                "retro_model_name": "pistachio_23Q3"
            }
        ],
        "use_fast_filter": True,
        "filter_threshold": 0.001,
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
        "return_reacting_atoms": False,
        "selectivity_check": False
    },
    "build_tree_options": {
        "expansion_time": 600,
        "max_chemicals": 5000,
        "max_branching": 25,
        "max_depth": 6,
        "exploration_weight": 1,
        "max_ppg_logic": "and",
        "max_ppg": 100,
        "return_first": False
    },
    "enumerate_paths_options": {
        "path_format": "json",
        "json_format": "nodelink",
        "sorting_metric": "number_of_reactions",
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
    with open("FDA_approved_filtered_19-23.csv", "r") as f:
        names_smiles = f.readlines()

    s = requests.Session()
    resp = s.post(
        url=f"http://{HOST}:{PORT}/api/admin/token",
        data={
            "username": USERNAME,
            "password": PASSWORD
        }
    )

    for i, name_smile in enumerate(names_smiles):
        # Skipping a problematic SMILES
        if i == 17:
            print(f"Skipping {name_smile}!!")
            continue

        data = copy.deepcopy(query_template)
        name, smi = name_smile.strip().split("\t")

        data["smiles"] = smi.strip()
        data["description"] = f"FDA_rerun_3, {smi.strip()}"

        resp = s.post(
            url=f"http://{HOST}:{PORT}/api/tree-search/mcts/call-async",
            json=data
        )
        print(f"i: {i}, SMILES: {smi.strip()}, result_id: {resp.text}")


if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = "9100"
    USERNAME = "askcos_admin"
    PASSWORD = "reallybadpassword"

    main()
