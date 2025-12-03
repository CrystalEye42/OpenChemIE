import io
import json
import pandas as pd
from fastapi import Depends, Form, HTTPException, Query, Response, status, UploadFile
from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Annotated, Any, Literal
from utils import register_util
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry


class BuyableProperty(BaseModel):
    name: str
    value: float | str

class BuyableInput(BaseModel):
    smiles: str
    ppg: float
    lead_time: str = ""
    source: str
    properties: list[dict] = []

class Buyable(BaseModel):
    id: str = Field(default=None, alias="_id")
    smiles: str = None
    ppg: float = 0.0
    lead_time: str = None
    source: str = ""
    properties: list[BuyableProperty] = None
    similarity: float = None


class PropertyCriteria(BaseModel):
    name: str
    logic: Literal[">", ">=", "<", "<=", "=="]
    value: float


class LookupInput(LowerCamelAliasModel):
    smiles: list[str]
    source: list[str] | str | None = None
    canonicalize: bool = True
    isomeric_smiles: bool = True


class LookupResponse(BaseModel):
    result: dict | None
    query: LookupInput


class PropertiesResponse(BaseModel):
    properties: list[str]


class SourcesResponse(BaseModel):
    sources: list[str]


class SearchInput(LowerCamelAliasModel):
    q: str = ""
    source: list[str] | str | None = None
    properties: list[PropertyCriteria] = None
    regex: bool = False
    sim_threshold: float = 1.0
    returnLimit: int = 100
    canonicalize: bool = True
    isomeric_smiles: bool = True
    similarity_method: str = "accurate"


class SearchResponse(BaseModel):
    result: list[Buyable] | None
    search: str


@register_util(name="buyables")
class Buyables:
    """Util class for Buyables, to be used as a wrapper around pricer"""
    prefixes = ["buyables"]
    methods_to_bind: dict[str, list[str]] = {
        "list_buyables": ["GET"],
        "retrieve": ["GET"],
        "create": ["POST"],
        "update": ["PUT"],
        "destroy": ["DELETE"],
        "search": ["POST"],
        "lookup": ["POST"],
        "sources": ["GET"],
        "properties": ["GET"],
        "upload": ["POST"]
    }

    def __init__(self, util_config: dict[str, Any]):
        pass

    def list_buyables(
        self,
        query_params: Annotated[SearchInput, Depends()],
        source: Annotated[list[str] | None, Query()] = None
    ) -> SearchResponse:
        # hardcode; Depends() doesn't fully work with list or union fields
        # https://github.com/tiangolo/fastapi/issues/5719
        query_params.source = source

        return self.search(query_params)

    @staticmethod
    def search(data: SearchInput) -> SearchResponse:
        pricer = get_util_registry().get_util(module="pricer")
        search_result = pricer.search(
            search_str=data.q,
            source=data.source,
            properties=data.properties,
            regex=data.regex,
            limit=data.returnLimit,
            canonicalize=data.canonicalize,
            isomeric_smiles=data.isomeric_smiles,
            sim_threshold=data.sim_threshold,
            similarity_method=data.similarity_method
        )
        resp = SearchResponse(result=search_result, search=data.q)

        return resp

    @staticmethod
    def retrieve(pk: str = None) -> Response:
        pricer = get_util_registry().get_util(module="pricer")

        resp = {"error": None, "result": None}

        try:
            result = pricer.get(pk)
        except NotImplementedError:
            resp["error"] = "Retrieving items by ID is not supported."

            return Response(
                content=json.dumps(resp),
                status_code=501,
                media_type="application/json"
            )

        if result:
            resp["result"] = result

            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )
        else:
            resp["error"] = "Requested item not found."

            return Response(
                content=json.dumps(resp),
                status_code=404,
                media_type="application/json"
            )

    @staticmethod
    def create(
        doc: Buyable,
        allowOverwrite: bool = True,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> Response:
        """Add new buyable entry"""
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)
        pricer = get_util_registry().get_util(module="pricer")

        if not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="create operation for buyables only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        resp = {
            "error": None,
            "success": False,
            "result": None,
            "inserted": False,
            "updated": False,
            "duplicate": False
        }

        try:
            result = pricer.add(doc.dict(), allow_overwrite=allowOverwrite)
        except NotImplementedError:
            resp["error"] = "Adding new items is not supported."

            return Response(
                content=json.dumps(resp),
                status_code=501,
                media_type="application/json"
            )

        if result["error"]:
            resp["error"] = result["error"]

            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )

        resp["success"] = True
        if result["doc"]:
            resp["result"] = result["doc"]
            if result["updated"]:
                resp["updated"] = True
            else:
                resp["inserted"] = True
        else:
            resp["duplicate"] = True

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )

    @staticmethod
    def update(
        pk: str,
        doc: Buyable,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> Response:
        """Update single buyables entry by mongo _id"""
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)
        pricer = get_util_registry().get_util(module="pricer")

        if not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="update operation for buyables only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        resp = {"error": None, "result": None}
        try:
            result = pricer.update(pk, doc.dict())
        except NotImplementedError:
            resp["error"] = "Updating items by ID is not supported."

            return Response(
                content=json.dumps(resp),
                status_code=501,
                media_type="application/json"
            )

        if result:
            resp["result"] = result

            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )
        else:
            resp["error"] = "Requested item not found."

            return Response(
                content=json.dumps(resp),
                status_code=404,
                media_type="application/json"
            )

    @staticmethod
    def destroy(
        pk: str = None,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> Response:
        """Delete a specific buyables entry from the database"""
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)
        pricer = get_util_registry().get_util(module="pricer")

        if not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="delete operation for buyables only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        resp = {"error": None, "success": False}

        try:
            success = pricer.delete(pk)
        except NotImplementedError:
            resp["error"] = "Deleting items by ID is not supported."

            return Response(
                content=json.dumps(resp),
                status_code=501,
                media_type="application/json"
            )

        if success:
            resp["success"] = True
        else:
            resp["error"] = "Could not delete requested item."

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )

    @staticmethod
    def sources() -> SourcesResponse:
        pricer = get_util_registry().get_util(module="pricer")
        resp = SourcesResponse(sources=pricer.list_sources())

        return resp

    @staticmethod
    def properties() -> PropertiesResponse:
        pricer = get_util_registry().get_util(module="pricer")
        resp = PropertiesResponse(properties=pricer.list_properties())

        return resp

    @staticmethod
    def lookup(data: LookupInput) -> LookupResponse:
        pricer = get_util_registry().get_util(module="pricer")
        result = pricer.lookup_smiles_list(
            smiles_list=data.smiles,
            source=data.source,
            canonicalize=data.canonicalize,
            isomeric_smiles=data.isomeric_smiles
        )
        resp = LookupResponse(result=result, query=data)

        return resp

    @staticmethod
    def upload(
        file: UploadFile,
        format: Annotated[str, Form()],
        returnLimit: Annotated[int, Form()],
        allowOverwrite: Annotated[bool, Form()],
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> Response:
        """API endpoint for uploading buyables data."""
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)
        pricer = get_util_registry().get_util(module="pricer")

        if not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="upload operation for buyables only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        resp = {
            "error": None,
            "inserted": [],
            "updated": [],
            "inserted_count": 0,
            "updated_count": 0,
            "duplicate_count": 0,
            "error_count": 0,
            "total_count": 0,
        }

        upload_file = file
        file_format = format
        return_limit = returnLimit
        allow_overwrite = allowOverwrite

        try:
            content = upload_file.file.read()
        except Exception as e:
            resp["error"] = f"Cannot read file: {e}!"

            return Response(
                content=json.dumps(resp),
                status_code=400,
                media_type="application/json"
            )

        if file_format == "json":
            try:
                upload_json = json.loads(content.decode("utf-8"))
            except Exception as e:
                resp["error"] = "Cannot parse json!"

                return Response(
                    content=json.dumps(resp),
                    status_code=400,
                    media_type="application/json"
                )
        
        elif file_format == "csv":
            try:
                df = pd.read_csv(io.StringIO(content.decode("utf-8")))
                upload_json = df.to_dict(orient="records")
            except Exception as e:
                resp["error"] = f"Cannot parse csv: {e}!"

                return Response(
                    content=json.dumps(resp),
                    status_code=400,
                    media_type="application/json"
                )
        else:
            resp["error"] = "File format not supported!"

            return Response(
                content=json.dumps(resp),
                status_code=400,
                media_type="application/json"
            )

        try:
            for doc in upload_json:
                BuyableInput(**doc)
        except Exception as e:
            resp["error"] = f"Some json fields are wrong; \n {e}"

            return Response(
                content=json.dumps(resp),
                status_code=400,
                media_type="application/json"
            )

        if (
            not isinstance(upload_json, list)
            or len(upload_json) == 0
            or not isinstance(upload_json[0], dict)
        ):
            resp["error"] = "Improperly formatted json!"

            return Response(
                content=json.dumps(resp),
                status_code=400,
                media_type="application/json"
            )

        try:
            result = pricer.add_many(
                new_docs=upload_json,
                allow_overwrite=allow_overwrite
            )
        except NotImplementedError:
            resp["error"] = "Adding new items is not supported."

            return Response(
                content=json.dumps(resp),
                status_code=501,
                media_type="application/json"
            )

        resp["error"] = result["error"]
        resp["inserted"] = result["inserted"][:return_limit]
        resp["updated"] = result["updated"][:return_limit]
        resp["inserted_count"] = result["inserted_count"]
        resp["updated_count"] = result["updated_count"]
        resp["duplicate_count"] = result["duplicate_count"]
        resp["error_count"] = result["error_count"]
        resp["total_count"] = result["total_count"]

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )
