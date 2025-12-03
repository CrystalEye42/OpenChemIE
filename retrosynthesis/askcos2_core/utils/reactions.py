import numpy as np
from configs import db_config
from pydantic import BaseModel
from pymongo import errors, MongoClient
from rdkit import Chem
from schemas.base import LowerCamelAliasModel
from typing import Any
from utils import register_util
from utils.similarity_search_utils import sim_search, sim_search_aggregate

DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_LEN = 2048


class ReactionsInput(LowerCamelAliasModel):
    ids: list[int | str]
    template_set: str = None


class ReactionsResponse(BaseModel):
    reactions: list


def _bits_to_array(bits: list[int], fp_size: int = 2048) -> np.ndarray:
    bits_array = np.zeros(fp_size, dtype=np.bool_)
    for bit in bits:
        bits_array[bit] = True

    return bits_array


@register_util(name="reactions")
class Reactions:
    """Util class for Reactions"""
    prefixes = ["reactions"]
    methods_to_bind: dict[str, list[str]] = {
        "post": ["POST"],
        "search_reaction_id": ["POST"],
        "lookup_by_exact_product_smiles": ["POST"],
        "lookup_similar_smiles": ["POST"]
    }

    def __init__(self, util_config: dict[str, Any]):
        self.client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)
        database = "askcos"
        collection = "reactions"
        # only products with valid reaction_smarts are kept in mol_collection
        mol_collection = "products_in_reactions"
        count_collection = "fp_counts_in_reactions"

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb for reactions")
        else:
            self.db = self.client[database]
            self.collection = self.db[collection]
            self.mol_collection = self.db[mol_collection]
            self.count_collection = self.db[count_collection]

    def lookup_by_exact_product_smiles(
        self,
        smiles: str,
        reaction_set: str = "USPTO_FULL"
    ) -> list[str]:
        """
        Lookup molecule entries (extracted from the reaction database) by exact match

        Returns:
            A list of reaction ids
        """

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []

        # proper canonicalization by removing atom mapping first
        for a in mol.GetAtoms():
            a.ClearProp("molAtomMapNumber")
            a.SetIsotope(0)

        canonical_smiles = Chem.MolToSmiles(mol)

        query = {
            "product_smiles": canonical_smiles,
            "template_set": reaction_set
        }
        cursor = self.mol_collection.find(query)
        reaction_ids = [str(mol["_id"]) for mol in cursor]

        return reaction_ids

    def lookup_similar_smiles(
        self,
        smiles: str,
        sim_threshold: float = 0.3,
        reaction_set: str = "USPTO_FULL",
        method: str = "accurate"
    ) -> list:
        """
        Lookup molecules in the database based on tanimoto similarity to the input
        SMILES string

        Note: assumes that a Mol Object, and Morgan Fingerprints are stored for
            each SMILES entry in the database

        Returns:
            A list of dictionary with one database entry for each molecule match including
            the tanimoto similarity to the query

        Note:
            Currently there are two options implemented lookup methods.
            The 'accurate' method is based on an aggregation pipeline in Mongo.
            The 'fast' method uses locality-sensitive hashing to greatly improve
            the lookup speed, at the cost of accuracy (especially at lower
            similarity thresholds).
        """
        query_mol = Chem.MolFromSmiles(smiles)
        if not query_mol:
            return []

        if method == "accurate":
            results = sim_search_aggregate(
                mol=query_mol,
                mol_collection=self.mol_collection,
                count_collection=self.count_collection,
                threshold=sim_threshold,
                reaction_set=reaction_set
            )
        elif method == "naive":
            results = sim_search(
                mol=query_mol,
                mol_collection=self.mol_collection,
                count_collection=self.count_collection,
                threshold=sim_threshold,
                reaction_set=reaction_set
            )
        elif method == "fast":
            raise NotImplementedError
        else:
            raise ValueError(f"Similarity search method '{method}' not implemented")

        output = [{'smiles': i['product_smiles'], 'tanimoto': i['tanimoto'], 'id': i['_id']} for i in results]

        return output

    def search_reaction_id(
        self,
        id: str,
        reaction_set: str = "USPTO_FULL"
    ) -> dict:
        """
        Lookup reaction collection using id.

        Returns:
            A dictionary containing document of reaction 
        """
        return self.collection.find_one({'_id': id, "template_set": reaction_set})

    def post(self, data: ReactionsInput) -> ReactionsResponse:
        query = {"reaction_id": {"$in": data.ids}}
        if data.template_set:
            # Processing for template subsets which use the same historian data
            query["template_set"] = data.template_set.split(":")[0]

        reactions_by_ids = list(self.collection.find(query))
        resp = ReactionsResponse(reactions=reactions_by_ids)

        return resp
