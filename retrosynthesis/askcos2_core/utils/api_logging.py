from configs import db_config
from fastapi import Depends, HTTPException, Response, status
from pydantic import BaseModel
from pymongo import errors, MongoClient
from typing import Annotated, Any
from utils import register_util
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry


class APILog(BaseModel):
    method: str
    count: dict[str, int]


@register_util(name="api_logging")
class APILogging:
    """
    Util class for API Logging
    """
    prefixes = ["api_logging"]
    methods_to_bind: dict[str, list[str]] = {
        "get": ["GET"],
        "purge": ["DELETE"]
    }

    def __init__(self, util_config: dict[str, Any] = None):
        self.client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)
        database = "logs"
        collection = "api_calls"

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb for api logging")
        else:
            self.db = self.client[database]
            self.collection = self.client[database][collection]

    def get(self, token: Annotated[str, Depends(oauth2_scheme)]
            ) -> list[APILog]:
        """
        API endpoint for accessing logs of api call counts

        Method: GET
        """
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        if not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="get operation for api logs only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        api_logs = self.collection.find()
        api_logs = [APILog(**log) for log in api_logs]
        print(api_logs)

        return api_logs

    def purge(self, token: Annotated[str, Depends(oauth2_scheme)]) -> Response:
        """
        API endpoint for purging logs of api call counts

        Method: GET
        """
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        if not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="purge operation for api logs only permitted by superusers",
                headers={"WWW-Authenticate": "Bearer"},
            )

        self.collection.delete_many({})

        return Response(content=f"Successfully purged the logs for all API calls!")
