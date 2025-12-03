from bson.objectid import ObjectId
from configs import db_config
from fastapi import Depends, Response
from pydantic import BaseModel
from pymongo import errors, MongoClient
from typing import Annotated, Any
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry


class BaseResponse(BaseModel):
    status_code: int
    message: str
    result: str | int | float | list | dict


class BaseBanlistController:
    """
    Base class for Banlist controller, alias for legacy BanlistViewSet
    """
    prefixes = []
    methods_to_bind: dict[str, list[str]] = {
        "get": ["GET"],
        "post": ["POST"],
        "delete": ["DELETE"],
        "activate": ["GET"],
        "deactivate": ["GET"],
    }

    def __init__(self, util_config: dict[str, Any]):
        """
        Initialize database connection.
        """
        assert util_config["engine"] == "db", \
            "Only the 'db' engine is supported for Banlist Controllers!"

        self.client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)
        database = util_config["database"]
        collection = util_config["collection"]

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb to load banlists")
        else:
            self.collection = self.client[database][collection]
            self.db = self.client[database]

    def get(self, token: Annotated[str, Depends(oauth2_scheme)]) -> BaseModel:
        """
        API endpoint for accessing banned entries.

        Method: GET

        Returns: list of banned entries belonging to the currently
            authenticated user
        """

        raise NotImplementedError

    def post(
        self,
        description: str = "no description",
        dt: str = None,
        smiles: str = None,
        active: bool = True,
        token: Annotated[str, Depends(oauth2_scheme)] = None
    ):
        """
        API endpoint for adding banned entries.

        Method: POST
        """
        raise NotImplementedError

    def delete(self, _id: str, token: Annotated[str, Depends(oauth2_scheme)]
               ) -> Response:
        """
        API endpoint to delete specified banlist entry by _id.

        Method: DELETE
        """

        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "user": user.username,
            "_id": ObjectId(_id)
        }
        self.collection.delete_one(query)

        return Response(content=f"Successfully delete banlist entry: {_id}!")

    def activate(self, _id: str, token: Annotated[str, Depends(oauth2_scheme)]
                 ) -> Response:
        """
        API endpoint to activate specified banlist entry by _id.

        Method: GET
        """

        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "user": user.username,
            "_id": ObjectId(_id)
        }
        self.collection.update_one(
            query,
            {"$set": {"active": True}}
        )

        return Response(content=f"Successfully activate banlist entry: {_id}!")

    def deactivate(self, _id: str, token: Annotated[str, Depends(oauth2_scheme)]
                   ) -> Response:
        """
        API endpoint to deactivate specified banlist entry by _id.

        Method: GET
        """

        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        query = {
            "user": user.username,
            "_id": ObjectId(_id)
        }
        self.collection.update_one(
            query,
            {"$set": {"active": False}}
        )

        return Response(content=f"Successfully deactivate banlist entry: {_id}!")
