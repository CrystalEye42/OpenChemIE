import json
from fastapi import Response
from typing import Any
from utils import register_util


@register_util(name="selectivity_refs")
class SelectivityRefs:
    """Util class for Selectivity references"""
    prefixes = ["selectivity/refs"]
    methods_to_bind: dict[str, list[str]] = {
        "get": ["GET"]
    }

    def __init__(self, util_config: dict[str, Any]):
        with open(util_config["file"], "r") as f:
            self.refs = json.load(f)

        self.refs = {int(ref["index"]): ref for ref in self.refs}

    def get(self, pk: str) -> Response:
        """Return references for a specific task index."""
        pk = int(pk)
        resp = {"index": pk, "error": None, "references": None}

        doc = self.refs.get(pk, None)

        if not doc:
            resp["error"] = f"Cannot find references for index {pk}"

            return Response(
                content=json.dumps(resp),
                status_code=404,
                media_type="application/json"
            )

        resp["references"] = doc["references"]

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )
