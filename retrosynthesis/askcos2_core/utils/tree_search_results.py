import json
import uuid
from configs import db_config
from datetime import datetime
from fastapi import Depends, HTTPException, Response
from pydantic import BaseModel, constr, Field
from pymongo import errors, MongoClient
from typing import Annotated, Any, Literal
from utils import register_util
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry
from utils.tree_search_results_util import standardize_result_ipp, standardize_result_tb

# TODO: fix the time zone issue


class TreeSearchSavedResults(BaseModel):
    """
    Reimplemented SavedResults model
    """
    user: str = None
    description: constr(max_length=1000) = None
    created: datetime = Field(default_factory=datetime.now)
    modified: datetime = Field(default_factory=datetime.now)
    dt: constr(max_length=200) = None
    result_id: constr(max_length=64) = None
    result: dict | None = None
    settings: dict | None = None
    tags: list[str] | str | None = Field(default_factory=list)
    check_date: str | None = None
    result_state: constr(max_length=64) = None
    result_type: str = "tree_builder"
    revision: int = 0

    target_smiles: str | None = None
    num_trees: int | None = 0

    public: bool = False
    shared_with: list[str] = Field(default_factory=list)

class UDS(BaseModel):
    node_dict: dict
    uuid2smiles: dict
    graph: dict
    pathways: list[dict]

class Metadata(BaseModel):
    uds: UDS

@register_util(name="tree_search_results_controller")
class TreeSearchResultsController:
    """
    Util class for tree search results controller
    """
    prefixes = ["results"]
    methods_to_bind: dict[str, list[str]] = {
        "list": ["GET"],
        "retrieve": ["GET"],
        "retrieve_pathway_metadata": ["POST"],
        "create": ["POST"],
        "update": ["PUT"],
        "destroy": ["DELETE"],
        "share": ["GET"],
        "unshare": ["GET"],
        "add": ["POST"],
        "remove": ["DELETE"]
    }

    def __init__(self, util_config: dict[str, Any]):
        """
        Initialize database connection.
        """
        self.client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)
        database = util_config["database"]
        collection = util_config["collection"]

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb to load results")
        else:
            self.collection = self.client[database][collection]
            self.db = self.client[database]

    def list(self, token: Annotated[str, Depends(oauth2_scheme)]
             ) -> list[TreeSearchSavedResults]:
        """
        API endpoint for accessing user results list.

        Method: GET

        Returns: list of tree builder results belonging to or shared with the currently
            authenticated user
        """
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "$or": [
                {"user": user.username},
                {
                    "public": True,
                    "shared_with": user.username
                }
            ]
        }
        # unsetting the result and settings fields to avoid gigantic returns
        cursor = self.collection.aggregate(
            [
                {"$match": query},
                {"$unset": ["result", "settings"]},
                {"$sort": {"dt": -1}}
            ]
        )
        results = []
        for r in cursor:
            result = TreeSearchSavedResults(**r)
            results.append(result)

        return results

    def retrieve_pathway_metadata(self, result: dict, pathways_properties_only=True):
        try:
            Metadata(**result)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"{e}"
            )
        result = standardize_result_tb(result)
        if pathways_properties_only:
            return result["uds"]["pathways_properties"]
        else:
            return result


    def retrieve(self, result_id: str, token: Annotated[str, Depends(oauth2_scheme)]
                 ) -> TreeSearchSavedResults:
        """
        API endpoint to retrieve specified result by result_id.

        Method: GET
        """

        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "result_id": result_id,
            "$or": [
                {"user": user.username},
                {
                    "public": True,
                    "shared_with": user.username
                }
            ]
        }
        result = self.collection.find_one(query)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Result with id {result_id} not found or not viewable!"
            )
        if result["result_state"] in ["completed", "ipp"]:
            result["_id"] = str(result["_id"])

            if result["result_type"] == "ipp":
                try:
                    result["result"] = standardize_result_ipp(result["result"])
                except:
                    # return results without formatting - frontend 
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unable to standardize ipp result for id: {result_id}!"
                    )
                # by pass tb result check
                return result

            elif result["result_type"] == "tree_builder":
                try:
                    result["result"] = standardize_result_tb(result["result"])
                except:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unable to standardize tb result for id: {result_id}!"
                    )

            result = TreeSearchSavedResults(**result)

            return result
        else:
            raise HTTPException(
                status_code=409,
                detail=f"Job not yet complete for id: {result_id}!"
            )

    def create(
        self,
        result: TreeSearchSavedResults,
        token: Annotated[str, Depends(oauth2_scheme)]
    ) -> Response:
        """
        API endpoint to create a result.

        Method: POST
        """
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        if not result.user:
            # This implies this is a create call from IPP
            result.user = user.username
            result.shared_with = [user.username]
            result.result_type = "ipp"
            result.result_state = "completed"

            if not result.result_id:
                result.result_id = str(uuid.uuid4())

        self.collection.insert_one(result.dict())

        resp = {
            "success": True,
            "id": result.result_id,
            "modified": result.modified.isoformat(timespec="milliseconds"),
            "message": f"Successfully create a result for user: {user.username}!"
        }

        return Response(
            content=json.dumps(resp),
            status_code=201,
            media_type="application/json"
        )

    def update(
        self,
        result_id: str,
        updated_result: dict,
        token: Annotated[str, Depends(oauth2_scheme)]
    ) -> Response:
        """
        API endpoint to update a result.

        Method: PUT
        """
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "result_id": result_id,
            "revision": updated_result["revision"] - 1,
            "$or": [
                {"user": user.username},
                {
                    "public": True,
                    "shared_with": user.username
                }
            ]
        }
        updated_result = {
            k: v for k, v in updated_result.items() if k in [
                "result", "settings", "description", "tags", "modified", "revision"
            ] and v
        }
        res = self.collection.update_one(
            query,
            {"$set": updated_result}
        )
        if not res.matched_count:
            resp = {
                "success": False,
                "error": f"Result {result_id} not editable by user {user.username}, "
                         f"or result changed since initial access!"
            }
            return Response(
                content=json.dumps(resp),
                status_code=404,
                media_type="application/json"
            )
        else:
            resp = {
                "success": True,
                "id": result_id,
                "modified": updated_result.get(
                    "modified", datetime.now()
                ).isoformat(timespec="milliseconds"),
                "message": f"Successfully update result {result_id}!"
            }
            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )

    def destroy(self, result_id: str, token: Annotated[str, Depends(oauth2_scheme)]
                ) -> Response:
        """
        API endpoint to delete specified result by result_id.

        Method: DELETE
        """

        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "user": user.username,
            "result_id": result_id
        }
        try:
            res = self.collection.delete_one(query)
        except Exception:
            resp = {
                "success": False,
                "message": f"Could not delete result {result_id}!"
            }
            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )
        if not res.deleted_count:
            resp = {
                "success": False,
                "message": f"Failed to delete result {result_id}!"
            }
            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )
        else:
            resp = {
                "success": True,
                "message": f"Successfully delete result: {result_id}!"
            }
            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )

    def share(self, result_id: str, token: Annotated[str, Depends(oauth2_scheme)]
              ) -> Response:
        """
        API endpoint to make public specific result by result_id.

        Method: GET
        """
        resp = {"id": result_id, "url": None, "error": None, "message": None}

        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "user": user.username,
            "result_id": result_id
        }
        res = self.collection.update_one(
            query,
            {"$set": {"public": True}}
        )
        if not res.matched_count:
            resp["error"] = f"Result {result_id} not editable by user {user.username}!"

            return Response(
                content=json.dumps(resp),
                status_code=401,
                media_type="application/json"
            )
        else:
            resp["url"] = f"/results?shared={result_id}"
            resp["message"] = f"Successfully make result public: {result_id}!"

            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )

    def unshare(self, result_id: str, token: Annotated[str, Depends(oauth2_scheme)]
                ) -> Response:
        """
        API endpoint to make private specific result by result_id.

        Method: GET
        """

        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "user": user.username,
            "result_id": result_id
        }
        res = self.collection.update_one(
            query,
            {"$set": {"public": False}}
        )
        if not res.matched_count:
            content = {
                "message": f"Result {result_id} not editable by user {user.username}!"
            }
            return Response(
                content=json.dumps(content),
                status_code=401,
                media_type="application/json"
            )
        else:
            return Response(content=f"Successfully make result private: {result_id}!")

    def add(self, result_id: str, token: Annotated[str, Depends(oauth2_scheme)]
            ) -> Response:
        """
        API endpoint to add specified result to the current user by result_id.

        Method: PUT
        """
        resp = {"id": result_id, "result": None, "error": None, "message": None}

        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "result_id": result_id,
            "public": True
        }
        res = self.collection.update_one(
            query,
            {"$push": {"shared_with": user.username}}
        )
        if not res.matched_count:
            resp["error"] = f"Result {result_id} not found or is not public!"

            return Response(
                content=json.dumps(resp),
                status_code=404,
                media_type="application/json"
            )
        else:
            resp["message"] = f"Successfully share result {result_id} for user: " \
                              f"{user.username}!"
            result = self.retrieve(result_id=result_id, token=token)
            resp["result"] = {
                "id": result.result_id,
                "state": result.result_state,
                "description": result.description,
                "created": result.created.strftime("%B %d, %Y %H:%M:%S %p UTC"),
                "modified": result.modified.strftime("%B %d, %Y %H:%M:%S %p UTC"),
                "type": result.result_type,
                "public": result.public,
                "tags": result.tags
            }

            if (
                    result.result_type == "tree_builder"
                    and result.result_state == "completed"
            ):
                resp["result"]["num_trees"] = result.num_trees

            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )

    def remove(self, result_id: str, token: Annotated[str, Depends(oauth2_scheme)]
               ) -> Response:
        """
        API endpoint to remove specified result to the current user by result_id.

        Method: DELETE
        """

        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "result_id": result_id,
            "public": True
        }
        res = self.collection.update_one(
            query,
            {"$pull": {"shared_with": user.username}}
        )
        if not res.matched_count:
            content = {
                "message": f"Result {result_id} not found or is not public!"
            }
            return Response(
                content=json.dumps(content),
                status_code=404,
                media_type="application/json"
            )
        else:
            return Response(content=f"Successfully unshare result {result_id} for "
                                    f"user: {user.username}!")

    def update_result_state(
        self,
        result_id: str,
        state: str,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> None:
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "user": user.username,
            "result_id": result_id
        }
        self.collection.update_one(
            query,
            {"$set": {"result_state": state}}
        )

    def save_results(
        self,
        result_id: str,
        result: BaseModel,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ) -> None:
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "user": user.username,
            "result_id": result_id
        }
        result_doc = result.dict()
        if "result_id" not in result_doc or not result_doc["result_id"]:
            result_doc["result_id"] = result_id

        self.collection.update_one(
            query,
            {"$set": {
                "result": result_doc,
                "num_trees": result_doc["stats"]["total_paths"]
            }}
        )
