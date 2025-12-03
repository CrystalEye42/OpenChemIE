import argparse
import multiprocessing
import numpy as np
import sys
import time
from configs import db_config
from pymongo import errors, MongoClient
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from typing import Any

RDLogger.DisableLog("rdApp.warning")

client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)
database = "askcos"
collection = "reactions"
mol_collection = "products_in_reactions"
index_collection = "index_for_reactions"
count_collection = "fp_counts_in_reactions"

try:
    client.server_info()
except errors.ServerSelectionTimeoutError:
    raise ValueError("Cannot connect to mongodb for reactions")
else:
    db = client[database]
    # force re-computing when invoked
    db.drop_collection(mol_collection)
    db.drop_collection(count_collection)

    collection = db[collection]
    mol_collection = db[mol_collection]
    index_collection = db[index_collection]
    count_collection = db[count_collection]


def parse_args():
    parser = argparse.ArgumentParser("precompute")
    parser.add_argument("--mode",
                        help="pre-computation mode",
                        choices=["reactions"], type=str, default="reactions")

    return parser.parse_args()


def _get_products_and_fps(rxn: dict) -> dict[str, Any] | None:
    fp_size = 2048

    reaction_smiles = rxn["reaction_smiles"].split()[0]
    reactant_smiles, _, product_smiles = reaction_smiles.split(">")
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return None
    reactant_mol = Chem.MolFromSmiles(reactant_smiles)
    if reactant_mol is None:
        return None

    product_mol = AllChem.RemoveHs(product_mol)

    mfp_bits = list(AllChem.GetMorganFingerprintAsBitVect(
        product_mol, radius=2, nBits=fp_size).GetOnBits())
    pfp_bits = list(Chem.rdmolops.PatternFingerprint(
        product_mol).GetOnBits())

    # mfp_bits = np.empty((1, fp_size), dtype=dtype)
    # pfp_bits = np.empty((1, fp_size), dtype=dtype)
    #
    # DataStructs.ConvertToNumpyArray(mfp_bits_vc, mfp_bits)
    # DataStructs.ConvertToNumpyArray(pfp_bits_vc, pfp_bits)
    #
    # mfp_bits = mfp_bits.astype(dtype=np.bool_)
    # pfp_bits = pfp_bits.astype(dtype=np.bool_)

    # proper canonicalization by removing atom mapping first
    for a in product_mol.GetAtoms():
        a.ClearProp("molAtomMapNumber")
        a.SetIsotope(0)
    product_smiles = Chem.MolToSmiles(product_mol)

    # only stores "indexes" and canonical smiles for mol
    # reaction metadata will be retrieved via rxn _id
    mol = {
        "_id": rxn["_id"],
        "template_set": rxn["template_set"],
        "product_smiles": product_smiles,
        "mfp_bits": mfp_bits,
        "mfp_count": len(mfp_bits),
        "pfp_bits": pfp_bits,
        "pfp_count": len(pfp_bits)
    }
    mol_collection.insert_one(mol)
    del mol

    mol = {
        "template_set": rxn["template_set"],
        "mfp_bits": mfp_bits
    }

    return mol


def precompute_fingerprints() -> None:
    start = time.time()
    mfp_counts = {}
    # all_mfp_bits = []
    success_count = 0

    p = multiprocessing.Pool()
    # query = {
    #     "reaction_smiles": {"$exists": True},
    #     "reaction_smarts": {"$nin": [None, ""]}
    # }
    query = {
        "reaction_smiles": {"$exists": True}
    }

    print("Precomputing fingerprints for all reactions (including newly added ones)")
    sys.stdout.flush()

    cursor = collection.find(query)
    for i, result in enumerate(p.imap_unordered(_get_products_and_fps, cursor)):
        if i > 0 and i % 100000 == 0:
            print(f"Processed {i} reactions with 'reaction_smiles' "
                  f"in {time.time() - start:.2f} seconds. "
                  f"Success count: {success_count}")
            sys.stdout.flush()

        if result is None:
            continue

        template_set = result["template_set"]
        mfp_bits = result["mfp_bits"]
        # all_mfp_bits.append(mfp_bits)

        if template_set not in mfp_counts:
            mfp_counts[template_set] = {}
        mfp_counts_by_template = mfp_counts[template_set]

        for bit in mfp_bits:
            mfp_counts_by_template[bit] = mfp_counts_by_template.get(bit, 0) + 1

        del result
        success_count += 1

    p.close()
    p.join()

    # print("Stacking all mfp_bits..")
    # all_mfp_bits = np.stack(all_mfp_bits)
    # print(f"Done stacking, shape: {all_mfp_bits.shape}, dtype: {all_mfp_bits.dtype}."
    #       f"Saving")
    #
    # ofn = "data/precompute/all_mfp_bits.npy"
    # np.save(ofn, all_mfp_bits)

    for template_set, mfp_counts_by_template in mfp_counts.items():
        for k, v in mfp_counts_by_template.items():
            count_collection.insert_one({
                "bit_index": k,
                "count": v,
                "template_set": template_set
            })

    mol_collection.create_index("product_smiles")
    # Additional indexing.. Also taking too long
    # mol_collection.create_index("mfp_bits")
    # mol_collection.create_index("mfp_count")
    # mol_collection.create_index("pfp_bits")
    # mol_collection.create_index("pfp_count")


def precompute_reactions():
    precompute_fingerprints()
    # precompute_string_fields()        # placeholder for future extension


def main():
    args = parse_args()
    if args.mode == "reactions":
        precompute_reactions()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
