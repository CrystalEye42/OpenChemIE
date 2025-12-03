from configs import db_config
from fastapi import Depends, HTTPException, Response, status
from pydantic import BaseModel
from pymongo import errors, MongoClient
from typing import Annotated, Any
from utils import register_util
from utils.registry import get_util_registry
from utils.oauth2 import oauth2_scheme
import os


class Config(BaseModel):
    key: str
    value: str


@register_util(name="frontend_config_controller")
class ConfigController:
    prefixes = ["frontend_config"]
    methods_to_bind: dict[str, list[str]] = {
        "get_all_config": ["GET"],
        "get_key": ["GET"],
    }

    def __init__(self, util_config: dict[str, Any]):
        """
        Initialize database connection.
        """
        assert util_config["engine"] == "db", \
            "Only the 'db' engine is supported for frontend config controller!"

        self.client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)

        database = util_config["database"]
        collection = util_config["collection"]

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb to load user data")
        else:
            self.collection = self.client[database][collection]
            self.db = self.client[database]

        configs = self.collection.find()
        configs = [Config(**c) for c in configs]
        config_keys = {config.key for config in configs}

        frontend_env_vars = {key: value for key, value in os.environ.items() if key.startswith("VITE")}
        for key, value in frontend_env_vars.items():
            if key not in config_keys:
                self.create(
                    key=key,
                    value=value
                )
            elif key in config_keys:
                self.update(
                    key=key,
                    value=value
                )

    def get_key(self, key: str) -> Config | None:
        query = {"key": key}
        try:
            config = self.collection.find_one(query)
            config = Config(**config)
        except:
            config = None

        return config

    def create(
        self,
        key: str,
        value: str
    ) -> Response:
        if not key.replace('_', '').isalnum():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid key: {key}! "
                       f"Only numbers, letters and underscores are allowed.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if self.collection.find_one({"key": key}):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"key: {key} already exists!",
                headers={"WWW-Authenticate": "Bearer"},
            )
        doc = {
            "key": key,
            "value": value
        }
        Config(**doc)         # data validation, if any
        self.collection.insert_one(doc)

        return Response(content=f"Successfully created key: {key}!")

    def update(
        self,
        key: str,
        value: str,
    ) -> Response:
        
        self.collection.update_one(
            {"key": key},
            {"$set": {
                "value": value,
            }}
        )

        return Response(content=f"Successfully update records for {key}!")

    def get_all_config(self
                      ) -> list[Config]:
        configs = self.collection.find()
        configs = [Config(**c) for c in configs]

        return configs
