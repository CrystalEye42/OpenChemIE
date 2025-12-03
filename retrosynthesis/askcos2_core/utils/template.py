import json
from bson import ObjectId
from configs import db_config
from fastapi import Response
from pymongo import errors, MongoClient
from schemas.base import LowerCamelAliasModel
from typing import Any
from utils import register_util


class TemplateInput(LowerCamelAliasModel):
    _id: str = None
    ids: list[str] = None
    field: str = None
    template_set: str = None


TEMPLATE_ALIASES = {
    "bkms": "bkms_metabolic",
    "pistachio:ringbreaker": "pistachio_ringbreaker",
    "reaxys_enzymatic": "reaxys_biocatalysis"
}


@register_util(name="template")
class Template:
    """Util class for Reactions"""
    prefixes = ["template"]
    methods_to_bind: dict[str, list[str]] = {
        "retrieve": ["GET"],
        "lookup": ["POST"],
        "sets": ["GET"]
    }

    def __init__(self, util_config: dict[str, Any]):
        self.client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)
        database = "askcos"
        collection = "retro_templates"

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb for templates")
        else:
            self.db = self.client[database]
            self.collection = self.client[database][collection]

    def find_one_by_id(self, _id: str) -> dict | None:
        transform = self.collection.find_one({"_id": _id})
        if not transform:
            try:
                transform = self.collection.find_one({"_id": ObjectId(_id)})
            except Exception:
                transform = None

        return transform

    def find_one_by_idx(self, template_idx: int, template_set: str) -> dict | None:
        transform = self.collection.find_one({
            "index": template_idx,
            "template_set": template_set
        })

        return transform

    def retrieve(self, pk: str) -> Response:
        """Return single template entry by mongo _id."""
        resp = {"error": None, "template": None}
        transform = self.find_one_by_id(_id=pk)

        if not transform:
            resp["error"] = f"Cannot find template with id {pk}"

            return Response(
                content=json.dumps(resp),
                status_code=404,
                media_type="application/json"
            )

        transform["_id"] = pk
        transform.pop("product_smiles", None)
        transform.pop("name", None)
        if transform.get("template_set") == "reaxys":
            # Each reaxys reaction ID may include multiple reaction references.
            # Condense them to the reaction ID only and omit the sub-index
            refs = transform.pop("references", [""])
            transform["references"] = list(dict.fromkeys(x.split("-")[0] for x in refs))
            transform["count"] = len(transform["references"])

        resp["template"] = transform

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )

    def lookup(self, data: TemplateInput) -> Response:
        """
        Lookup a list of template IDs.
        If field is specified, returns the field value for each template.
        Otherwise, returns the full document for each template.

        Method: POST

        Parameters:

        - `ids` (list[str]): list of template IDs to look up
        - `field` (str, optional): specific field to return
        - `template_set` (str, optional): template set to search

        Returns:

        - `query` (dict): parameters provided in request
        - `result` (dict): result dictionary keyed by template ID
        """
        ids = data.ids
        field = data.field
        template_set = data.template_set

        query = {"_id": {"$in": ids}}
        if template_set is not None:
            query["template_set"] = template_set

        proj = None
        if field is not None:
            proj = [field]

        cursor = self.collection.find(filter=query, projection=proj)
        result = {str(doc.pop("_id")): doc[field] if field else doc
                  for doc in cursor}
        resp = {"result": result, "query": data.dict()}

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )

    def sets(self) -> Response:
        """Returns available template sets that exist in mongodb"""
        names = self.collection.distinct("template_set")
        attributes = {}
        for name in names:
            alias = TEMPLATE_ALIASES.get(name, name)
            attributes[alias] = list(
                self.collection
                .find_one({"template_set": name}, {"attributes": 1})
                .get("attributes", {})
                .keys()
            )

        aliases = [TEMPLATE_ALIASES.get(name, name) for name in names]
        resp = {"template_sets": aliases, "attributes": attributes}

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )
