import json
import random
from configs import db_config
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pymongo import errors, MongoClient
from typing import Any
from utils import register_util


@register_util(name="cache_controller")
class CacheController:
    """Util class for controlling the cache"""
    prefixes = []
    methods_to_bind: dict[str, list[str]] = {}

    def __init__(self, util_config: dict[str, Any] = None):
        self.cache_maps = {}
        self.cache_size = 10000

        self.client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)
        database = "cache"

        try:
            self.client.server_info()
        except errors.ServerSelectionTimeoutError:
            raise ValueError("Cannot connect to mongodb for cache")
        else:
            # reset the db from the Mongo side as well
            self.client.drop_database(database)
            self.db = self.client[database]

    def get(self, module_name: str, input: BaseModel) -> dict:
        # any KeyError will be handled at the respective wrapper
        cache_map = self.cache_maps[module_name]
        input_hash = json.dumps(jsonable_encoder(input))
        _id = cache_map[input_hash]

        try:
            collection = self.db[module_name]
            response = collection.find_one(_id)
            del response["_id"]
        except:     # weird bug; may be due to lag for async mongo deletion?
            raise KeyError

        return response

    def add(self, module_name: str, input: BaseModel, response: BaseModel) -> None:
        if module_name not in self.cache_maps:
            self.cache_maps[module_name] = {}
        cache_map = self.cache_maps[module_name]
        input_hash = json.dumps(jsonable_encoder(input))

        collection = self.db[module_name]
        _id = collection.insert_one(response.dict()).inserted_id
        cache_map[input_hash] = _id

        while len(cache_map) >= self.cache_size:
            k = random.choice(list(cache_map.keys()))
            collection.delete_one({"_id": cache_map[k]})
            del cache_map[k]
