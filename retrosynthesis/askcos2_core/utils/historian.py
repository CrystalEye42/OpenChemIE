import gzip
import hashlib
import json
import os
from bson import ObjectId
from configs import db_config
from pymongo import errors, MongoClient
from rdkit import Chem
from typing import Any
from utils import register_util


@register_util(name="historian")
class Historian:
    """
    Util class for Historian, to be used as a controller (over Mongo/FileHistorian)
    """
    prefixes = ["historian"]
    methods_to_bind: dict[str, list[str]] = {
        "lookup_smiles": ["POST"],
        "lookup_smiles_list": ["POST"],
        "search": ["POST"],
        "list_template_sets": ["GET"],
        "get": ["GET"],
    }

    def __init__(self, util_config: dict[str, Any]):
        engine = util_config["engine"]
        if engine == "db":
            self._historian = MongoHistorian(
                config=db_config.MONGO,
                database=util_config["database"],
                collection=util_config["collection"]
            )
            self.collection = self._historian.collection
            self._estimated_count = None
            self._template_sets = []
        elif engine == "file":
            self._historian = FileHistorian(files=util_config["files"])
        else:
            raise ValueError(f"Unsupported historian engine: {engine}! "
                             f"Only 'db' or 'file' is supported")

    @staticmethod
    def canonicalize(smiles: str, isomeric_smiles: bool = True):
        """
        Canonicalize the input SMILES.

        Returns:
            str: canonicalized SMILES or empty str on failure
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
        except Exception:
            return ""
        else:
            return smiles

    @staticmethod
    def hash_smiles(smiles: str) -> str:
        """
        Generate modified md5 hash of input SMILES string.
        """
        return str(int(hashlib.md5(smiles.encode("utf-8")).hexdigest(), 16))

    def lookup_smiles(
        self,
        smiles: str,
        template_sets: list[str] = None,
        canonicalize=True,
        isomeric_smiles=True
    ) -> dict[str, int]:
        """
        Looks up number of occurrences by SMILES.

        Tries it as-entered and then re-canonicalizes it in RDKit unless the
        user specifies that the string is definitely already canonical.

        Args:
            smiles (str): SMILES string of molecule to look up.
            template_sets (list, optional): Template sets to consider when
                aggregating statistics.
            canonicalize (bool, optional): whether to canonicalize SMILES string
            isomeric_smiles (bool, optional): whether to generate isomeric
                SMILES string when performing canonicalization

        Returns:
            dict: containing 'as_reactant' and 'as_product' keys
        """
        if canonicalize:
            smiles = self.canonicalize(
                smiles=smiles,
                isomeric_smiles=isomeric_smiles
            ) or smiles
        hashed_smiles = self.hash_smiles(smiles)

        return self._historian.lookup_smiles(
            smiles=smiles,
            hashed_smiles=hashed_smiles,
            template_sets=template_sets
        )

    def lookup_smiles_list(
        self,
        smiles_list: list[str],
        template_sets: list[str] = None,
        canonicalize: bool = True,
        isomeric_smiles: bool = True
    ) -> dict[str, dict[str, int]]:
        results = {}
        for smiles in smiles_list:
            results[smiles] = self.lookup_smiles(
                smiles=smiles,
                template_sets=template_sets,
                canonicalize=canonicalize,
                isomeric_smiles=isomeric_smiles
            )

        return results

    # The following methods are Mongo only
    def search(
        self,
        smiles: str,
        template_sets: list[str] = None,
        canonicalize: bool = True,
        isomeric_smiles: bool = True
    ) -> list:
        """
        Search the database based on the specified criteria.

        Args:
            smiles (str): SMILES string of molecule to look up.
            template_sets (list, optional): Template sets to consider when
                aggregating statistics.
            canonicalize (bool, optional): whether to canonicalize SMILES string
            isomeric_smiles (bool, optional): whether to generate isomeric
                SMILES string when performing canonicalization

        Returns:
            list: full documents of all records matching the criteria
        """

        assert isinstance(self._historian, MongoHistorian), \
            f"search() is only implemented for MongoHistorian"

        if canonicalize:
            smiles = self.canonicalize(
                smiles=smiles,
                isomeric_smiles=isomeric_smiles
            ) or smiles

        hashed_smiles = self.hash_smiles(smiles)

        # Processing for template subsets which use the same historian data
        if template_sets:
            template_sets = list({ts.split(":")[0] for ts in template_sets})

        query = {"smiles": {"$in": [smiles, hashed_smiles]}}
        if template_sets:
            query["template_set"] = {"$in": template_sets}

        search_result = list(self.collection.find(query))

        for doc in search_result:
            doc["_id"] = str(doc["_id"])

        return search_result

    def list_template_sets(self) -> list:
        """
        Retrieve all available template sets.

        Returns:
            list: list of source names
        """

        assert isinstance(self._historian, MongoHistorian), \
            f"list_template_sets() is only implemented for MongoHistorian"

        estimated_count = self.collection.estimated_document_count()
        if self._estimated_count is None or self._estimated_count != estimated_count:
            self._template_sets = list(self.collection.distinct("template_set"))
            self._estimated_count = estimated_count
        return self._template_sets

    def get(self, _id: str) -> dict:
        """
        Get a single entry by its _id.
        """

        assert isinstance(self._historian, MongoHistorian), \
            f"get() is only implemented for MongoHistorian"

        result = self.collection.find_one({"_id": ObjectId(_id)})
        if result and result.get("_id"):
            result["_id"] = str(result["_id"])

        return result


class MongoHistorian:
    def __init__(self, config: dict, database: str, collection: str):
        """
        Initialize database connection.
        """
        self.client = MongoClient(serverSelectionTimeoutMS=1000, **config)

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb to load historian")
        else:
            self.collection = self.client[database][collection]
            self.db = self.client[database]

    def lookup_smiles(
        self,
        smiles: str,
        hashed_smiles: str,
        template_sets: list[str] = None
    ) -> dict[str, int]:
        # Processing for template subsets which use the same historian data
        default_result = {"as_reactant": 0, "as_product": 0}

        if template_sets:
            template_sets = list({ts.split(":")[0] for ts in template_sets})

        query = {"smiles": {"$in": [smiles, hashed_smiles]}}
        if template_sets:
            query["template_set"] = {"$in": template_sets}
        output = list(
            self.collection.aggregate(
                [
                    {"$match": query},
                    {
                        "$group": {
                            "_id": None,
                            "as_reactant": {"$sum": "$as_reactant"},
                            "as_product": {"$sum": "$as_product"},
                        }
                    },
                ]
            )
        )
        if output:
            result = output[0]
            del result["_id"]
        else:
            result = default_result

        return result


class FileHistorian:
    def __init__(self, files: list[str]):
        """
        Load data from local file.
        """
        print("Loading chemhistorian from file...")

        self.data = {}
        for filename in files:
            if not os.path.isfile(filename):
                raise ValueError(f"Chemical data file does not exist: {filename}")

            with gzip.open(filename, "rt", encoding="utf-8") as f:
                data = json.load(f)

            for entry in data:
                smiles = entry.pop("smiles")
                template_set = entry.pop("template_set")
                self.data.setdefault(template_set, {})[smiles] = entry

        print("Historian is fully loaded.")

    def lookup_smiles(
        self,
        smiles: str,
        hashed_smiles: str,
        template_sets: list[str] = None
    ) -> dict[str, int]:
        default_result = {"as_reactant": 0, "as_product": 0}

        # Processing for template subsets which use the same historian data
        if template_sets:
            template_sets = list({ts.split(":")[0] for ts in template_sets})

        template_sets = template_sets or self.data.keys()
        results = [default_result]
        for template_set in template_sets:
            results.append(self.data.get(template_set, {}).get(smiles))
            results.append(self.data.get(template_set, {}).get(hashed_smiles))

        return {
            "as_reactant": sum(
                doc["as_reactant"] for doc in results if doc is not None
            ),
            "as_product": sum(doc["as_product"] for doc in results if doc is not None),
        }
