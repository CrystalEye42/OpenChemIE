import os
import pandas as pd
from bson import Binary, ObjectId
from configs import db_config
from fastapi import Query
from pymongo import errors, MongoClient
from pymongo.collection import ReturnDocument
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Annotated, Any
from utils import register_util
import hashlib
from utils.similarity_search_utils import sim_search_aggregate_buyables, sim_search_buyables


@register_util(name="pricer")
class Pricer:
    """Util class for Pricer, to be used as a controller (over Mongo/FilePricer)"""
    prefixes = ["pricer"]
    methods_to_bind: dict[str, list[str]] = {
        "lookup_smarts": ["POST"],
        "lookup_smiles": ["POST"]
    }
    # methods_to_bind: dict[str, list[str]] = {
    #     "lookup_smiles": ["POST"],
    #     "lookup_smiles_list": ["POST"],
    #     "lookup_smarts": ["POST"],
    #     "search": ["POST"],
    #     "list_sources": ["GET"],
    #     "list_properties": ["GET"],
    #     "get": ["GET"],
    #     "add": ["POST"],
    #     "add_many": ["POST"],
    #     "update": ["POST"],
    #     "delete": ["DELETE"]
    # }

    def __init__(self, util_config: dict[str, Any]):
        engine = util_config["engine"]
        if engine == "db":
            self._pricer = MongoPricer(
                config=db_config.MONGO,
                database=util_config["database"],
                collection=util_config["collection"]
            )
            self.collection = self._pricer.collection
        elif engine == "file":
            self._pricer = FilePricer(
                path=util_config["file"],
                precompute_mols=util_config["precompute_mols"]
            )
        else:
            raise ValueError(f"Unsupported pricer engine: {engine}! "
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

    def lookup_smiles(
        self,
        smiles: str,
        source: Annotated[list[str] | None, Query()] = None,
        canonicalize: bool = True,
        isomeric_smiles: bool = True
    ) -> dict | None:
        """
        Lookup data for the requested SMILES, based on lowest price.

        Args:
            smiles (str): SMILES string to look up
            source (list or str, optional): buyables sources to consider;
                if ``None`` (default), include all sources, otherwise
                must be single source or list of sources to consider;
            canonicalize (bool, optional): whether to canonicalize SMILES string
            isomeric_smiles (bool, optional): whether to generate isomeric
                SMILES string when performing canonicalization

        Returns:
            dict: data for the requested SMILES, None if not found
        """
        if canonicalize:
            smiles = self.canonicalize(
                smiles=smiles,
                isomeric_smiles=isomeric_smiles
            ) or smiles

        return self._pricer.lookup_smiles(smiles=smiles, source=source)

    def lookup_smiles_list(
        self,
        smiles_list: list[str],
        source: list[str] | str | None = None,
        canonicalize: bool = True,
        isomeric_smiles: bool = True
    ) -> dict | None:
        """
        Lookup data for a list of SMILES, based on lowest price for each.

        SMILES not found in database are omitted from the output dict.

        Args:
            smiles_list (list): list of SMILES strings to look up
            source (list or str, optional): buyables sources to consider;
                if ``None`` (default), include all sources, otherwise
                must be single source or list of sources to consider;
            canonicalize (bool, optional): whether to canonicalize SMILES string
            isomeric_smiles (bool, optional): whether to generate isomeric
                SMILES string when performing canonicalization

        Returns:
            dict: mapping from input SMILES to data dict
        """
        if canonicalize:
            smiles_list = [
                self.canonicalize(smi, isomeric_smiles=isomeric_smiles) or smi
                for smi in smiles_list
            ]

        return self._pricer.lookup_smiles_list(smiles_list=smiles_list, source=source)

    def lookup_smarts(
        self,
        smarts: str,
        limit: int | None = None,
        precomputed_mols: bool = False
    ) -> list | dict:
        return self._pricer.lookup_smarts(
            smarts=smarts,
            limit=limit,
            precomputed_mols=precomputed_mols
        )

    # The following methods are Mongo only
    def search(
        self,
        search_str: str,
        source: list[str] | str | None = None,
        properties: list[dict[str, Any]] = None,
        regex: bool = False,
        sim_threshold: float = 1.0,
        limit: int = 100,
        canonicalize: bool = True,
        isomeric_smiles: bool = True,
        similarity_method: str = "accurate",
    ) -> list:
        """
        Search the database based on the specified criteria.

        Returns:
            list: full documents of all buyables matching the criteria
        """

        assert isinstance(self._pricer, MongoPricer), \
            f"search() is only implemented for MongoPricer"

        query = {}
        tanimoto_similarities = None
        keys_to_keep = [
            "_id", "smiles", "ppg", "lead_time", "source", "properties", "tanimoto"
        ]
        db_comparison_map = {
            ">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "==": "$eq"
        }

        if search_str:
            if regex:
                smarts_lookup_res = self.lookup_smarts(smarts=search_str, limit=limit)
                smiles_matches = list([d["smiles"] for d in smarts_lookup_res])[:limit]
                query["smiles"] = {"$in": smiles_matches}
            elif sim_threshold == 1:
                if canonicalize:
                    search_str = (
                        self.canonicalize(search_str, isomeric_smiles=isomeric_smiles)
                        or search_str
                    )
                query["smiles"] = search_str
            else:
                similarity_results = self._pricer.lookup_similar_smiles(
                    smiles=search_str,
                    sim_threshold=sim_threshold,
                    method=similarity_method,
                    limit=limit
                )
                tanimoto_similarities = {
                    r["smiles"]: r["tanimoto"] for r in similarity_results
                }
                smiles_matches = list(tanimoto_similarities.keys())
                query["smiles"] = {"$in": smiles_matches}

        if source is not None:
            query["source"] = {"$in": self._pricer._source_to_query(source)}

        if properties is not None:
            property_query = []
            for item in properties:
                property_query.append(
                    {
                        "properties": {
                            "$elemMatch": {
                                "name": item["name"],
                                "value": {
                                    db_comparison_map[item["logic"]]: item["value"]
                                },
                            }
                        }
                    }
                )
            query["$and"] = property_query

        search_result = list(
            self.collection.find(query, projection=keys_to_keep).limit(limit)
        )

        for doc in search_result:
            doc["_id"] = str(doc["_id"])
            if tanimoto_similarities:
                doc["tanimoto"] = "{:.2f}".format(tanimoto_similarities[doc["smiles"]])

        if tanimoto_similarities:
            search_result = sorted(
                search_result, key=lambda x: x["tanimoto"], reverse=True
            )

            def key_switch(item):
                tanimoto = item["tanimoto"]
                item["similarity"] = tanimoto
                del item["tanimoto"]
                return item
            search_result = list(map(key_switch, search_result))
            # print(search_result)

        for res in search_result:
            properties= res.get("properties")
            if properties is None: properties = []
            new_properties = []            
            for prop in properties:
                key, value = list(prop.items()).pop()
                new_properties.append(
                    {"name": key,
                     "value": value}
                )
            res["properties"] = new_properties
            if res.get("similarity") is None:
                res["similarity"] = str(sim_threshold)
        
        return search_result

    def list_sources(self) -> list[str]:
        """
        Retrieve all available source names.

        Returns:
            list: list of source names
        """

        assert isinstance(self._pricer, MongoPricer), \
            f"list_sources() is only implemented for MongoPricer"

        sources = [s for s in self.collection.distinct("source") if s]
        if (
            self.collection.find_one(filter={"source": {"$in": [None, ""]}})
            is not None
        ):
            sources.append("none")

        return sources

    def list_properties(self) -> list[str]:
        """
        Retrieve all available property names.

        Note: Not all documents may have all properties defined.

        Returns:
            list: list of property names
        """

        assert isinstance(self._pricer, MongoPricer), \
            f"list_properties() is only implemented for MongoPricer"

        return list(self.collection.distinct("properties.name"))

    def get(self, _id: str) -> dict:
        """
        Get a single entry by its _id.
        """

        assert isinstance(self._pricer, MongoPricer), \
            f"get() is only implemented for MongoPricer"

        # result = self.collection.find_one({"_id": ObjectId(_id)})
        result = self.collection.find_one({"_id": _id})
        if result and result.get("_id"):
            result["_id"] = str(result["_id"])

        return result

    def update(self, _id: str, new_doc: dict) -> dict:
        """
        Update a single entry by its _id.
        """

        assert isinstance(self._pricer, MongoPricer), \
            f"update() is only implemented for MongoPricer"

        result = self.collection.find_one_and_replace(
            {"_id": ObjectId(_id)}, new_doc, return_document=ReturnDocument.AFTER
        )
        if result and result.get("_id"):
            result["_id"] = str(result["_id"])

        return result

    def delete(self, _id: str) -> bool:
        """
        Delete a single entry by its _id.
        """

        assert isinstance(self._pricer, MongoPricer), \
            f"delete() is only implemented for MongoPricer"

        delete_result = self.collection.delete_one({"_id": _id})
        # delete_result = self.collection.delete_one({"_id": ObjectId(_id)})

        return delete_result.deleted_count > 0

    def add(self, new_doc: dict, allow_overwrite: bool = True) -> dict:
        """
        Add a new entry to the database.
        """

        assert isinstance(self._pricer, MongoPricer), \
            f"add() is only implemented for MongoPricer"

        new_doc["smiles"] = self.canonicalize(new_doc["smiles"])
        smi, source = new_doc["smiles"], new_doc["source"]
        smi_vendor = f"{smi}{source}"
        hash_id = hashlib.sha256(smi_vendor.encode('utf-8')).hexdigest()
        new_doc["_id"] = hash_id
        result = {"doc": None, "updated": False, "error": None}
        query = {
            "smiles": new_doc["smiles"],
            "source": new_doc["source"],
        }
        existing_doc = self.collection.find_one(query)
        if existing_doc:
            if allow_overwrite:
                replace_result = self.collection.replace_one(query, new_doc)
                if replace_result.matched_count:
                    new_doc["_id"] = str(existing_doc["_id"])
                    result["doc"] = new_doc
                    result["updated"] = True
                else:
                    result["error"] = "Failed to update buyable entry."
        else:
            insert_result = self.collection.insert_one(new_doc)
            if insert_result.inserted_id:
                new_doc["_id"] = str(insert_result.inserted_id)
                result["doc"] = new_doc
            else:
                result["error"] = "Failed to add buyable entry."

        return result

    def add_many(self, new_docs: list[dict], allow_overwrite: bool = True) -> dict:
        """
        Add a list of new entries to the database.
        """

        assert isinstance(self._pricer, MongoPricer), \
            f"add_many() is only implemented for MongoPricer"

        result = {
            "error": None,
            "inserted": [],
            "updated": [],
            "inserted_count": 0,
            "updated_count": 0,
            "duplicate_count": 0,
            "error_count": 0,
            "total_count": len(new_docs),
        }

        for new_doc in new_docs:
            res = self.add(new_doc, allow_overwrite=allow_overwrite)
            if not res["error"]:
                if res["doc"]:
                    if res["updated"]:
                        result["updated"].append(res["doc"])
                        result["updated_count"] += 1
                    else:
                        result["inserted"].append(res["doc"])
                        result["inserted_count"] += 1
                else:
                    result["duplicate_count"] += 1
            else:
                result["error"] = res["error"]
                result["error_count"] += 1

        return result


class MongoPricer:
    def __init__(self, config: dict, database: str, collection: str):
        """
        Initialize database connection.
        """
        self.client = MongoClient(serverSelectionTimeoutMS=1000, **config)

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb to load prices")
        else:
            self.collection = self.client[database][collection]
            self.db = self.client[database]

        self.smarts_query_index = {}
        self.count_collection = None

    @staticmethod
    def _source_to_query(source: list[str] | str | None) -> list[str] | None:
        """
        Convert no source keyword to query for MongoDB.

        Args:
            source (str or list): source names, possibly including 'none'

        Returns:
            list: modified source list replacing 'none' with None and ''
        """
        if source is not None:
            if not isinstance(source, list):
                source = [source]
            if "none" in source:
                # Include both null and empty string source in query
                source.remove("none")
                source.extend([None, ""])

        return source

    def lookup_smiles(
        self,
        smiles: str,
        source: list[str] | str | None = None
    ) -> dict | None:
        if source == []:
            # If no sources are allowed, there is no need to perform lookup
            # Empty list is checked explicitly here, since None means source
            # will not be included in query, and '' is a valid source value
            return None

        if self.collection is not None:
            query = {"smiles": smiles}

            if source is not None:
                query["source"] = {"$in": self._source_to_query(source)}

            cursor = self.collection.find(query)
            result = min(cursor, key=lambda x: x["ppg"], default=None)
            if result:
                result["_id"] = str(result["_id"])
                # keeping only these fields. Once the mols are computed, serialization
                # becomes an issue.
                result = {
                    k: v for k, v in result.items() if k in [
                        "_id", "smiles", "ppg", "lead_time", "source", "properties"
                    ]
                }
            return result
        else:
            return None

    def lookup_smiles_list(
        self,
        smiles_list: list[str],
        source: list[str] | str | None = None
    ) -> dict[str, Any]:
        query = {"smiles": {"$in": smiles_list}}

        if source is not None:
            query["source"] = {"$in": self._source_to_query(source)}

        cursor = self.collection.aggregate(
            [
                {"$match": query},
                {"$sort": {"smiles": 1, "ppg": 1}},
                {
                    "$group": {
                        "_id": "$smiles",
                        "ppg": {"$first": "$ppg"},
                        "source": {"$first": "$source"},
                    }
                },
            ]
        )
        result = {}
        for doc in cursor:
            result[str(doc.pop("_id"))] = {
                k: v for k, v in doc.items()
                if k in ["smiles", "ppg", "lead_time", "source", "properties"]
            }

        return result

    def is_mols_precomputed(self) -> bool:
        query = {"mol": {"$ne": None}}
        result = self.collection.find_one(query)
        if result:
            return True
        else:
            return False

    def precompute_mols(self, batch_size: int = 10000) -> None:
        """
        Stores rdkit Mol objects as a Binary,a molecular fingerprint and bit
        counts, and a pattern fingerprint and bit counts for each molecule in
        the database
        """
        print(f"{self.collection.count_documents(filter={})} "
              f"documents in the buyables database")
        idxs = [s["_id"] for s in self.collection.find({}, {"_id": 1}) if s]
        full_batch_idx = int(len(idxs) / batch_size) * batch_size
        if batch_size > len(idxs):
            batched_idxs = [idxs]
        else:
            splits = list(range(0, full_batch_idx, batch_size)) + [len(idxs)]
            batched_idxs = [idxs[i:j] for i, j in zip(splits[:-1], splits[1:])]

        document_list = []
        print(f"Precomputing mols in {len(batched_idxs)} batches")
        mfp_counts = {}
        for i, batch in enumerate(batched_idxs):
            query = {"_id": {"$in": batch}}
            documents = self.collection.find(query)
            for document in documents:
                rdmol = Chem.MolFromSmiles(document["smiles"])
                mfp = list(
                    AllChem.GetMorganFingerprintAsBitVect(
                        rdmol, 2, nBits=2048
                    ).GetOnBits()
                )
                pfp = list(Chem.rdmolops.PatternFingerprint(rdmol).GetOnBits())
                document["mol"] = Binary(rdmol.ToBinary())
                document["mfp"] = {"bits": mfp, "count": len(mfp)}
                document["pfp"] = {"bits": pfp, "count": len(pfp)}
                document_list.append(document)

                for bit in mfp:
                    mfp_counts[bit] = mfp_counts.get(bit, 0) + 1
            self.collection.delete_many(query)

            print(f"Done computing RDK mol objects for batch "
                  f"{i} out of {len(batched_idxs)}")
            self.collection.insert_many(document_list)
            document_list = []

        print(f"{self.collection.count_documents(filter={})} "
              f"documents in the buyables database")

        self.count_collection = self.db["count_collection"]
        self.count_collection.delete_many({})
        print(f"{self.count_collection.count_documents(filter={})} "
              f"documents in the counts database")
        for k, v in mfp_counts.items():
            self.count_collection.insert_one({"_id": k, "count": v})
        self.collection.create_index("mfp.bits")
        self.collection.create_index("mfp.count")
        self.collection.create_index("pfp.bits")
        self.collection.create_index("pfp.count")

        print("Created new indexes in the database")

    def lookup_smarts(
        self,
        smarts: str,
        limit: int | None = None,
        precomputed_mols: bool = False
    ) -> list:
        """
        Lookup molecules in the database using a SMARTS pattern string

        Note: assumes that a Mol Object and pattern fingerprints are stored for
            each SMILES entry in the database

        Returns:
            A dictionary with one database entry for each molecule match

        Note:
            Implementation adapted from https://github.com/rdkit/mongo-rdkit/blob
                /master/mongordkit/Search/substructure.py
        """
        if not self.is_mols_precomputed():
            self.precompute_mols()

        if smarts in self.smarts_query_index.keys():
            matched_ids = self.smarts_query_index[smarts]
            query = {"smiles": {"$in": matched_ids}}
            cursor = self.collection.aggregate(
                [
                    {"$match": query},
                    {
                        "$group": {
                            "_id": "$_id",
                            "smiles": {"$first": "$smiles"},
                        }
                    },
                ]
            )
            # result = list(result)
            result = []
            for doc in cursor:
                trimmed_doc = {
                    "_id": str(doc["_id"]),
                    "smiles": doc["smiles"]
                }
                result.append(trimmed_doc)

        else:
            pattern = Chem.MolFromSmarts(smarts)
            query_fp = list(Chem.rdmolops.PatternFingerprint(pattern).GetOnBits())
            qfp_len = len(query_fp)
            matched_ids = []
            query = {
                "mol": {"$ne": None},
                "pfp.count": {"$gte": qfp_len},
                "pfp.bits": {"$all": query_fp},
            }

            cursor = self.collection.aggregate(
                [
                    {"$match": query},
                    {
                        "$group": {
                            "_id": "$smiles",
                            "smiles": {"$first": "$smiles"},
                            "mol": {"$first": "$mol"},
                        }
                    },
                ]
            )

            # Perform substructure matching
            result = []
            for doc in cursor:
                try:
                    rdmol = Chem.Mol(doc["mol"])
                    if rdmol.HasSubstructMatch(pattern, useChirality=True):
                        matched_ids.append(doc["_id"])
                        # Not returning the "mol"; serialization issue with fastapi
                        trimmed_doc = {
                            "_id": str(doc["_id"]),
                            "smiles": doc["smiles"]
                        }
                        result.append(trimmed_doc)
                except KeyError as e:
                    print("Key error {}, {}".format(e, doc["smiles"]))

            self.smarts_query_index[smarts] = matched_ids

        if limit:
            result = result[:limit]

        return result

    def lookup_similar_smiles(
        self,
        smiles: str,
        sim_threshold: float,
        limit: int = None,
        method: str = "accurate"
    ) -> list:
        """
        Lookup molecules in the database based on tanimoto similarity to the input
        SMILES string

        Note: assumes that a Mol Object, and Morgan Fingerprints are stored for
            each SMILES entry in the database

        Returns:
            A dictionary with one database entry for each molecule match including
            the tanimoto similarity to the query

        Note:
            Currently there are two options implemented lookup methods.
            The 'accurate' method is based on an aggregation pipeline in Mongo.
            The 'fast' method uses locality-sensitive hashing to greatly improve
            the lookup speed, at the cost of accuracy (especially at lower
            similarity thresholds).
        """
        if not self.is_mols_precomputed():
            self.precompute_mols()

        query_mol = Chem.MolFromSmiles(smiles)
        if method == "accurate":
            results = sim_search_aggregate_buyables(
                query_mol,
                self.collection,
                None,
                sim_threshold,
            )
        elif method == "naive":
            results = sim_search_buyables(
                query_mol,
                self.collection,
                None,
                sim_threshold,
            )
        elif method == "fast":
            raise NotImplementedError
        else:
            raise ValueError(f"Similarity search method '{method}' not implemented")

        results = sorted(results, key=lambda x: x["tanimoto"], reverse=True)
        if limit:
            results = results[:limit]

        return results


class FilePricer:
    def __init__(self, path: str, precompute_mols: bool = False):
        """
        Load price data from local file.
        """
        if os.path.isfile(path):
            self.path = path
            self.data = pd.read_json(
                path,
                orient="records",
                dtype={"smiles": "object", "source": "object", "ppg": "float"},
                compression="gzip",
            )
            print(f"Loaded prices from flat file: {path}")
            self.indexed_queries = {}
        else:
            print(f"Buyables file does not exist: {path}")

        if precompute_mols:
            self.data["mols"] = [Chem.MolFromSmiles(x) for x in self.data["smiles"]]

        self.smarts_query_index = {}

    def lookup_smiles(
        self,
        smiles: str,
        source: list[str] | str | None = None
    ) -> dict | None:
        if source == []:
            # If no sources are allowed, there is no need to perform lookup
            # Empty list is checked explicitly here, since None means source
            # will not be included in query, and '' is a valid source value
            return None

        if self.data is not None:
            query = self.data["smiles"] == smiles

            if source is not None:
                if isinstance(source, list):
                    query = query & (self.data["source"].isin(source))
                else:
                    query = query & (self.data["source"] == source)

            results = self.data.loc[query]
            if len(results.index):
                idxmin = results["ppg"].idxmin()
                return results.loc[idxmin].to_dict()
            else:
                return None
        else:
            return None

    def lookup_smiles_list(
        self,
        smiles_list: list[str],
        source: list[str] | str | None = None
    ):
        raise NotImplementedError

    def lookup_smarts(
        self,
        smarts: str,
        limit: int | None = None,
        precomputed_mols: bool = False
    ) -> dict:
        if smarts in self.smarts_query_index.keys():
            matches = self.smarts_query_index[smarts]

        elif precomputed_mols:
            pattern = Chem.MolFromSmarts(smarts)
            matches = self.data["mols"].apply(lambda x: x.HasSubstructMatch(pattern))
            self.smarts_query_index[smarts] = matches

        else:
            pattern = Chem.MolFromSmarts(smarts)
            matches = self.data["smiles"].apply(
                lambda x: Chem.MolFromSmiles(x).HasSubstructMatch(pattern)
            )
            self.smarts_query_index[smarts] = matches

        return self.data[matches].to_dict(orient="records")
