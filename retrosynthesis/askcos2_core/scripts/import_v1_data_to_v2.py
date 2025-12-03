import json
import os
import traceback as tb
from bson import ObjectId
from configs import db_config
from datetime import datetime
from pymongo import errors, MongoClient
from tqdm import tqdm


client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)
try:
    client.server_info()
except errors.ServerSelectionTimeoutError:
    raise ValueError("Cannot connect to mongodb for cache")


def insert_one_if_not_found(collection, username: str, fields: dict) -> None:
    query = {
        "user": username,
        "smiles": fields["smiles"]
    }
    if not collection.find_one(query):
        doc = {
            "user": username,
            "description": fields["description"],
            "created": datetime.strptime(fields["created"], "%Y-%m-%dT%H:%M:%S.%fZ"),
            "dt": fields["dt"],
            "smiles": fields["smiles"],
            "active": fields["active"],
        }
        collection.insert_one(doc)


def convert_result(result: dict, username: str, fields: dict) -> dict:
    result["user"] = username
    result["description"] = fields["description"]
    result["created"] = fields["created"]
    result["modified"] = fields["created"]
    result["dt"] = fields["dt"]
    result["result_id"] = result["_id"]

    result["result_type"] = fields["result_type"]
    result["revision"] = 0
    result["public"] = False
    result["shared_with"] = [username]

    if "settings" in result:
        settings = result["settings"]
        if result["result_type"] == "tree_builder":
            result["target_smiles"] = settings["smiles"]
            result["result_state"] = "completed"
            try:
                result["num_trees"] = result["result"]["stats"]["total_paths"]
            except KeyError:
                # some super legacy results
                result["num_trees"] = None
            new_settings = {
                "expand_one_options": {
                    "template_max_count": settings.get("template_count", 1000),
                    "template_max_cum_prob": settings.get("max_cum_template_prob", 0.995),
                    "banned_chemicals": settings.get("forbidden_molecules", list()),
                    "banned_reactions": settings.get("known_bad_reactions", list()),
                    "retro_backend_options": [
                        {
                            "retro_backend": "template_relevance",
                            "retro_model_name": p.get("template_set", "unknown"),
                            "max_num_templates": settings.get("template_count", 1000),
                            "max_cum_prob": settings.get("max_cum_template_prob", 0.995),
                            "attribute_filter": p.get("attribute_filter", list())
                        } for p in settings.get("template_prioritizers", list())
                    ],
                    "use_fast_filter": True,
                    "filter_threshold": settings.get("filter_threshold", 0.0),
                    "retro_rerank_backend": "relevance_heuristic",
                    "cluster_precursors": False,
                    "extract_template": False,
                    "return_reacting_atoms": True,
                    "selectivity_check": False
                },
                "build_tree_options": {
                    "expansion_time": settings.get("expansion_time", 60),
                    "max_iterations": settings.get("max_iterations"),
                    "max_chemicals": settings.get("max_chemicals"),
                    "max_reactions": settings.get("max_reactions"),
                    "max_templates": settings.get("max_templates"),
                    "max_branching": settings.get("max_branching"),
                    "max_depth": settings.get("max_depth"),
                    "exploration_weight": 1,
                    "return_first": settings.get("return_first", False),
                    "max_trees": settings.get("max_trees", 500),
                    "buyable_logic": "and",
                    "max_ppg_logic": "none",
                    "max_ppg": 0,
                    "max_scscore_logic": "none",
                    "max_scscore": 0,
                    "chemical_property_logic": "none",
                    "max_chemprop_c": 0,
                    "max_chemprop_n": 0,
                    "max_chemprop_o": 0,
                    "max_chemprop_h": 0,
                    "chemical_popularity_logic": "none",
                    "min_chempop_reactants": 5,
                    "min_chempop_products": 5,
                    "buyables_source": settings.get("buyables_source"),
                    "custom_buyables": list()
                },
                "enumerate_paths_options": {
                    "path_format": "json",
                    "json_format": settings.get("json_format", "nodelink"),
                    "sorting_metric": "plausibility",
                    "validate_paths": True,
                    "score_trees": settings.get("score_trees", False),
                    "cluster_trees": settings.get("cluster_trees", False),
                    "cluster_method": settings.get("cluster_method", "hdbscan"),
                    "min_samples": settings.get("min_samples", 5),
                    "min_cluster_size": settings.get("min_cluster_size", 5),
                    "paths_only": False,
                    "max_paths": settings.get("max_trees", 500)
                }
            }
            result["settings"] = new_settings

        elif result["result_type"] == "ipp":
            result["result_state"] = "ipp"

            tb_settings = settings["tb"]
            interactive_path_planner_settings = {
                "retro_backend_options": [
                    {
                        "retro_backend": "template_relevance",
                        "retro_model_name": p.get("template_set", "unknown"),
                        "attribute_filter": p.get("attribute_filter", list()),
                        "max_num_templates": 1000,
                        "max_cum_prob": 0.999
                    } for p in tb_settings.get("templatePrioritizers", list())
                ],
                "banned_chemicals": [],
                "banned_reactions": [],
                "use_fast_filter": True,
                "fast_filter_threshold": tb_settings.get("minPlausibility", 1e-6),
                "retro_rerank_backend": "relevance_heuristic",
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
                "selectivity_check": False,
                "group_by_strategy": False
            }

            tree_builder_settings = {}
            settings.update({
                "interactive_path_planner": interactive_path_planner_settings,
                "tree_builder": tree_builder_settings
            })

    return result


def share_result(collection, result_id: str, username: str):
    query = {
        "result_id": result_id
    }
    collection.update_one(
        query,
        {
            "$set": {"public": True},
            "$push": {"shared_with": username}
        }
    )


def main():
    mysql_dump = os.environ["MYSQL_DUMP_PATH_IN_CONTAINER"]
    mongo_dump = os.environ["MONGO_DUMP_PATH_IN_CONTAINER"]
    assert mysql_dump.endswith(".json")
    assert mongo_dump.endswith(".json")

    print(f"Loading mysql objects from {mysql_dump}")
    with open(mysql_dump, "r") as f:
        mysql_objects = json.load(f)
    print("Loaded")

    print(f"Loading mongo docs from {mongo_dump}")
    with open(mongo_dump, "r") as f:
        mongo_docs = json.load(f)
    print("Loaded")

    print("Indexing mongo docs with _id")
    mongo_docs_keyed = {doc["_id"]: doc for doc in tqdm(mongo_docs)}

    banned_chemicals = []
    banned_reactions = []
    saved_results = []
    result_ids = set()
    result_ids_invalid = set()

    # create user map; only pk -> username is really needed
    print("Creating user_map pk -> username from mysql_dump..")
    user_map = {}
    for o in mysql_objects:
        if o["model"] == "auth.user":
            pk = o["pk"]
            username = o["fields"]["username"]
            user_map[pk] = username
        elif o["model"] == "main.blacklistedchemicals":
            banned_chemicals.append(o)
        elif o["model"] == "main.blacklistedreactions":
            banned_reactions.append(o)
        elif o["model"] == "main.savedresults":
            saved_results.append(o)

    # migrate banned_chemicals
    collection = client["askcos"]["banned_chemicals"]
    print("Migrating blasklisted chemicals")
    for o in tqdm(banned_chemicals):
        user_pk = o["fields"]["user"]
        username = user_map[user_pk]
        insert_one_if_not_found(
            collection=collection,
            username=username,
            fields=o["fields"]
        )

    # migrate banned_reactions
    collection = client["askcos"]["banned_reactions"]
    print("Migrating blasklisted reactions")
    for o in tqdm(banned_reactions):
        user_pk = o["fields"]["user"]
        username = user_map[user_pk]
        insert_one_if_not_found(
            collection=collection,
            username=username,
            fields=o["fields"]
        )

    # migrate saved_results
    collection = client["results"]["results"]
    print("Migrating saved results")
    saved_results = sorted(saved_results, key=lambda _d: _d["pk"])
    for o in tqdm(saved_results):
        user_pk = o["fields"]["user"]
        username = user_map[user_pk]
        result_id = o["fields"]["result_id"]

        if result_id in result_ids:
            print(f"result_id {result_id} found in result_ids!")
            print(f"Sharing {result_id} with user {username}")
            share_result(
                collection=collection,
                result_id=result_id,
                username=username
            )
        elif result_id in result_ids_invalid:
            continue
        else:
            # by right the first occurance of the result should be associated with the owner
            try:
                result = mongo_docs_keyed[result_id]
            except KeyError:
                result_ids_invalid.add(result_id)
                continue

            if "result" not in result:
                # some weird result
                result_ids_invalid.add(result_id)
                continue

            try:
                result = convert_result(
                    result=result,
                    username=username,
                    fields=o["fields"]
                )
                # force purge
                collection.delete_one({"result_id": result["result_id"]})
                collection.insert_one(result)
            except:
                result_ids_invalid.add(result_id)
                continue

            result_ids.add(result_id)

    print(f"Number of invalid result_ids: {len(result_ids_invalid)}")


if __name__ == "__main__":
    main()
