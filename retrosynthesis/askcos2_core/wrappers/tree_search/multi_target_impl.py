import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from typing import Any


def get_best_paths(
    all_paths: dict[str, list[dict[str, Any]] | None]
) -> dict[str, dict[str, Any] | None]:
    best_buyables_size = 10

    # compute leaves (the buyables) for all paths
    for target, paths in all_paths.items():
        # print(target)
        if not paths:
            continue

        for path in paths:
            path["leaves"] = [
                node["smiles"] for node in path["nodes"]
                if node["type"] == "chemical" and node["terminal"]
            ]
            # print(path["leaves"])

    # compute the best sets of buyables
    running_best_buyables = []
    for target, paths in all_paths.items():
        if not paths:
            continue

        new_best_buyables = [path["leaves"] for path in paths]
        new_best_buyables = new_best_buyables[:best_buyables_size]

        if not running_best_buyables:
            running_best_buyables = new_best_buyables
        else:
            running_best_buyables = _get_running_best_buyables(
                running_best_buyables=running_best_buyables,
                new_best_buyables=new_best_buyables
            )
        # print(f"running_best_buyables: {running_best_buyables}")

    # compute the best paths (containing buyables most similar to best buyables)
    best_paths = {}
    for target, paths in all_paths.items():
        if not paths:
            best_paths[target] = None
            continue

        highest_path_similarity = 0.0
        best_path = None

        for path in paths:
            path_similarity = _get_path_similarity(
                leaves=path["leaves"],
                running_best_buyables=running_best_buyables
            )
            if path_similarity > highest_path_similarity:
                highest_path_similarity = path_similarity
                best_path = path
        # print(best_path)

        best_paths[target] = best_path

    return best_paths


def _get_running_best_buyables(
    running_best_buyables: list[list[str]],
    new_best_buyables: list[list[str]]
) -> list[list[str]]:
    scores_existing = [
        _get_pairwise_score(existing_buyable, new_best_buyables)
        for existing_buyable in running_best_buyables
    ]

    scores_new = [
        _get_pairwise_score(new_buyable, running_best_buyables)
        for new_buyable in new_best_buyables
    ]

    scores_agg = scores_existing + scores_new
    buyables_agg = running_best_buyables + new_best_buyables

    top_ids = np.argsort(scores_agg)[::-1]
    new_running_best_buyables = [buyables_agg[i] for i in top_ids]

    return new_running_best_buyables


def _get_pairwise_score(
    ref_buyable: list[str],
    compared_buyables: list[list[str]]
) -> float:
    score = 0.0
    ref_mols = [Chem.MolFromSmiles(smi) for smi in ref_buyable]
    ref_fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        for mol in ref_mols
    ]

    for compared_buyable in compared_buyables:
        compared_mols = [Chem.MolFromSmiles(smi) for smi in compared_buyable]
        compared_fps = [
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            for mol in compared_mols
        ]

        tanimoto = max(
            max(DataStructs.BulkTanimotoSimilarity(ref_fp, compared_fps))
            for ref_fp in ref_fps
        )
        if tanimoto > score:
            score = tanimoto

    return score


def _get_path_similarity(
    leaves: list[str],
    running_best_buyables: list[list[str]]
) -> float:
    path_similarity = _get_pairwise_score(
        ref_buyable=leaves,
        compared_buyables=running_best_buyables
    )

    return path_similarity
