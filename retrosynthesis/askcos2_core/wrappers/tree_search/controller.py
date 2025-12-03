import re
import uuid
from datetime import datetime
from fastapi import Depends
from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Annotated, Any, Literal
from utils.oauth2 import oauth2_scheme
from utils.registry import get_util_registry
from utils.tree_search_results import TreeSearchSavedResults
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper
from wrappers.registry import get_wrapper_registry
from wrappers.tree_search.mcts import (
    MCTSInput as TreeSearchMCTSInput,
    MCTSResponse as TreeSearchMCTSResponse
)
from wrappers.tree_search.retro_star import (
    ExpandOneOptions,
    BuildTreeOptions,
    EnumeratePathsOptions,
    RetroStarInput as TreeSearchRetroStarInput,
    RetroStarResponse as TreeSearchRetroStarResponse
)


class TreeSearchInput(LowerCamelAliasModel):
    backend: Literal[
        "mcts", "retro_star"
    ] = Field(
        default="mcts",
        description="backend for tree search"
    )
    smiles: str = Field(
        description="target SMILES for tree search",
        example="CN(C)CCOC(c1ccccc1)c1ccccc1"
    )
    description: str | None = Field(
        default="",
        description="description of the tree search task",
    )
    tags: str | None = Field(
        default="",
        description="tags of the tree search task",
    )
    expand_one_options: ExpandOneOptions = Field(
        default_factory=ExpandOneOptions,
        description="options for one-step expansion"
    )
    build_tree_options: BuildTreeOptions = Field(
        default_factory=BuildTreeOptions,
        description="options for tree search"
    )
    enumerate_paths_options: EnumeratePathsOptions = Field(
        default_factory=EnumeratePathsOptions,
        description="options for path enumeration once the tree is built"
    )
    run_async: bool = False
    result_id: str = str(uuid.uuid4())


class UDS(BaseModel):
    node_dict: dict
    uuid2smiles: dict
    graph: list[dict]
    pathways: list[list[dict]]
    pathways_properties: list[dict]


class TreeSearchResult(BaseModel):
    stats: dict[str, Any] | None
    uds: UDS | None
    version: int | str | None = 2
    result_id: str = ""


class TreeSearchOutput(BaseModel):
    placeholder: str


class TreeSearchResponse(BaseResponse):
    result: TreeSearchResult | None


@register_wrapper(
    name="tree_search_controller",
    input_class=TreeSearchInput,
    output_class=TreeSearchOutput,
    response_class=TreeSearchResponse
)
class TreeSearchController(BaseWrapper):
    """Tree Search Controller"""
    prefixes = ["tree_search/controller", "tree_search"]
    methods_to_bind: dict[str, list[str]] = {
        "get_config": ["GET"],
        "get_doc": ["GET"],
        "call_sync": ["POST"],
        "call_sync_without_token": ["POST"],
        "call_async": ["POST"],
        "retrieve": ["GET"]
    }
    backend_wrapper_names = {
        "mcts": "tree_search_mcts",
        "retro_star": "tree_search_retro_star"
    }

    def __init__(self):
        pass        # TODO: proper inheritance

    def call_sync(
        self,
        input: TreeSearchInput,
        token: Annotated[str, Depends(oauth2_scheme)]
    ) -> TreeSearchResponse:
        """
        Endpoint for synchronous call to the tree search controller,
        which dispatches the call to respective tree search backend service
        """
        module = self.backend_wrapper_names[input.backend]
        wrapper = get_wrapper_registry().get_wrapper(module=module)

        wrapper_input = self.convert_input(
            input=input,
            backend=input.backend
        )
        wrapper_response = wrapper.call_sync(
            wrapper_input,
            token=token
        )
        response = self.convert_response(
            wrapper_response=wrapper_response,
            backend=input.backend
        )

        return response

    def call_sync_without_token(self, input: TreeSearchInput) -> TreeSearchResponse:
        """
        Endpoint for synchronous call to the tree search controller.
        Skip login at the expense of losing access to user banned lists.
        """
        module = self.backend_wrapper_names[input.backend]
        wrapper = get_wrapper_registry().get_wrapper(module=module)

        wrapper_input = self.convert_input(
            input=input,
            backend=input.backend
        )
        wrapper_response = wrapper.call_sync_without_token(
            wrapper_input
        )
        response = self.convert_response(
            wrapper_response=wrapper_response,
            backend=input.backend
        )

        return response

    async def call_async(
        self,
        input: TreeSearchInput,
        token: Annotated[str, Depends(oauth2_scheme)],
        priority: int = 0
    ) -> str:
        """
        Endpoint for asynchronous call to the tree search controller,
        which dispatches the call to respective tree search backend service
        """
        user_controller = get_util_registry().get_util(module="user_controller")
        user = user_controller.get_current_user(token)

        input.run_async = True
        input.result_id = str(uuid.uuid4())
        # Note that we can't use task_id as the result_id,
        # as it needs to be known beforehand
        settings = input.dict()
        settings = {k: v for k, v in settings.items() if "option" in k}

        saved_results = TreeSearchSavedResults(
            user=user.username,
            target_smiles=input.smiles,
            description=input.description,
            created=datetime.now(),
            modified=datetime.now(),
            result_id=input.result_id,
            result_state="pending",
            result_type="tree_builder",
            result=None,
            settings=settings,
            tags=[tag.strip() for tag in re.split(r"[,;]", input.tags)],
            shared_with=[user.username]
        )
        results_controller = get_util_registry().get_util(
            module="tree_search_results_controller"
        )
        results_controller.create(result=saved_results, token=token)

        from askcos2_celery.tasks import tree_search_task
        async_result = tree_search_task.apply_async(
            args=(self.name, input.dict(), token), priority=priority)
        task_id = async_result.id

        return task_id

    async def retrieve(self, task_id: str) -> TreeSearchResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_input(
        input: TreeSearchInput, backend: str
    ) -> TreeSearchMCTSInput | TreeSearchRetroStarInput:
        wrapper_input = input.dict()
        del wrapper_input["backend"]

        if backend == "mcts":
            # use_value_network is a Retro*-only field
            # all other fields are the same, for now
            del wrapper_input["build_tree_options"]["use_value_network"]
            wrapper_input = TreeSearchMCTSInput(**wrapper_input)
        elif backend == "retro_star":
            wrapper_input = TreeSearchRetroStarInput(**wrapper_input)
        else:
            raise ValueError(f"Unsupported tree search backend: {backend}!")

        return wrapper_input

    @staticmethod
    def convert_response(
        wrapper_response:
            TreeSearchMCTSResponse | TreeSearchRetroStarResponse,
        backend: str
    ) -> TreeSearchResponse:
        status_code = wrapper_response.status_code
        message = wrapper_response.message
        if backend in ["mcts", "retro_star"]:
            # trivial conversion to be future-proof,
            # in case there are more tree builders
            result = wrapper_response.result
        else:
            raise ValueError(f"Unsupported tree search backend: {backend}!")

        response = {
            "status_code": status_code,
            "message": message,
            "result": result
        }
        response = TreeSearchResponse(**response)

        return response
