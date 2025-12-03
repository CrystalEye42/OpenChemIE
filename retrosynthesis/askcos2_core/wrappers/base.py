import requests
from celery.result import AsyncResult
from configs import db_config
from datetime import date
from pydantic import BaseModel
from pymongo import errors, MongoClient


class BaseResponse(BaseModel):
    status_code: int
    message: str
    result: str | int | float | list | dict


class BaseWrapper:
    input_class: type[BaseModel]
    output_class: type[BaseModel]
    response_class: type[BaseResponse]

    name: str = "base"
    prefixes: list[str] = None
    methods_to_bind: dict[str, list[str]] = {
        "get_config": ["GET"],
        "get_doc": ["GET"],
        "call_sync": ["POST"],
        "call_async": ["POST"],
        "retrieve": ["GET"]
    }
    methods_to_log: list[str] = [
        "call_raw",
        "call_sync"
    ]

    logs_client = MongoClient(serverSelectionTimeoutMS=1000, **db_config.MONGO)
    try:
        logs_client.server_info()
    except errors.ServerSelectionTimeoutError:
        print("Cannot connect to mongodb for api logging")
        pass
    else:
        logs_db = logs_client["logs"]
        logs_collection = logs_client["logs"]["api_calls"]

    def __init__(self, config: dict):
        self.config = config
        self.prediction_url = config["deployment"].get("custom_prediction_url", "")
        if not self.prediction_url:
            self.prediction_url = config["deployment"]["default_prediction_url"]

        self.config["prediction_url_in_use"] = self.prediction_url
        self.session_sync = requests.Session()

    def get_config(self) -> dict:
        return self.config

    def get_doc(self) -> str:
        return self.__doc__

    def is_ready(self) -> bool:
        try:
            resp = self.session_sync.get(self.prediction_url)
            status_code = resp.status_code
            if status_code != 404:
                return True
            # 404 can also be root endpoint not found rather than unavailable service
            # Right now we only need to check FastAPI (via "detail")
            # or torchserve (via "message")
            resp = resp.json()
            ready = (
                resp.get("detail") == "Not Found" or
                resp.get("message") ==
                "Requested resource is not found, please refer to API document."
            )
        except:
            ready = False

        return ready

    def call_raw(self, input: BaseModel) -> BaseModel:
        response = self.session_sync.post(
            self.prediction_url,
            json=input.dict(),
            timeout=self.config["deployment"]["timeout"]
        )
        output = response.json()
        output = self.output_class(**output)

        return output

    def call_sync(self, input: BaseModel) -> BaseResponse:
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: BaseModel, priority: int = 0) -> str:
        from askcos2_celery.tasks import base_task
        async_result = base_task.apply_async(
            args=(self.name, input.dict()), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> BaseResponse | None:
        result = AsyncResult(task_id)
        response = result.result
        response = self.response_class(**response)

        return response

    @staticmethod
    def convert_output_to_response(output: BaseModel) -> BaseResponse:
        response = {
            "status_code": 200,
            "message": "",
            "result": output.results
        }
        response = BaseResponse(**response)

        return response

    def __getattribute__(self, item: str):
        # Hook for logging function (thus API) calls
        _methods_to_log = super().__getattribute__("methods_to_log")
        if item in _methods_to_log:
            try:
                _collection = super().__getattribute__("logs_collection")
                _wrapper_name = super().__getattribute__("name")
                query = {"method": f"{_wrapper_name}.{item}"}
                dt = date.today().strftime("%y%m%d")
                _collection.update_one(
                    query,
                    {"$inc": {f"count.{dt}": 1}},
                    upsert=True
                )
            except AttributeError:
                # logs_collection might not have been created for weird mongo issue
                # for now simply skip logging for robustness
                pass

        return super().__getattribute__(item)
