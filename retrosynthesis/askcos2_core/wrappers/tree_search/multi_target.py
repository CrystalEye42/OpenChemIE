import uuid
from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from typing import Any
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper
from wrappers.registry import get_wrapper_registry
from wrappers.tree_search.mcts import (
    BuildTreeOptions,
    EnumeratePathsOptions,
    ExpandOneOptions,
    MCTSInput,
    MCTSResponse
)
from wrappers.tree_search.multi_target_impl import get_best_paths


class MultiTargetInput(LowerCamelAliasModel):
    targets: list[str] = Field(
        default_factory=list,
        description="list of target SMILES for MCTS tree building",
        example=[
            "CN(C)CCOC(c1ccccc1)c1ccccc1",
            "CN(C)CCCOC(c1ccccc1)c1ccccc1"
        ]
    )
    description: str | None = Field(
        default="",
        description="description of the MCTS task",
    )
    tags: str | None = Field(
        default="",
        description="tags of the MCTS task",
    )
    expand_one_options: ExpandOneOptions = Field(
        default_factory=ExpandOneOptions,
        description="options for one-step expansion"
    )
    build_tree_options: BuildTreeOptions = Field(
        default_factory=BuildTreeOptions,
        description="options for MCTS tree search"
    )
    enumerate_paths_options: EnumeratePathsOptions = Field(
        default_factory=EnumeratePathsOptions,
        description="options for path enumeration once the tree is built"
    )
    run_async: bool = False
    result_id: str = str(uuid.uuid4())


class MultiTargetOutput(BaseModel):
    placeholder: str


class MultiTargetResponse(BaseResponse):
    result: dict[str, dict[str, Any] | None] | None


@register_wrapper(
    name="tree_search_multi_target",
    input_class=MultiTargetInput,
    output_class=MultiTargetOutput,
    response_class=MultiTargetResponse
)
class MultiTargetWrapper(BaseWrapper):
    """Wrapper class for Multi Target MCTS"""
    prefixes = ["tree_search/multi_target"]
    methods_to_bind: dict[str, list[str]] = {
        "call_sync_without_token": ["POST"]
    }

    def __init__(self):
        pass        # TODO: proper inheritance

    def call_sync_without_token(self, input: MultiTargetInput) -> MultiTargetResponse:
        """
        Endpoint for synchronous call to the multi target tree searcher.
        Skip login at the expense of losing access to user banned lists.
        """
        wrapper = get_wrapper_registry().get_wrapper(module="tree_search_mcts")

        all_paths = {}
        for target in input.targets:
            wrapper_input = self.convert_input(
                target=target,
                multi_target_input=input
            )
            wrapper_response = wrapper.call_sync_without_token(
                wrapper_input
            )

            paths = self.get_paths_from_response(wrapper_response)
            all_paths[target] = paths

        best_paths = get_best_paths(all_paths)
        response = self.convert_output_to_response(best_paths)

        return response

    @staticmethod
    def convert_input(
        target: str,
        multi_target_input: MultiTargetInput
    ) -> MCTSInput:
        wrapper_input = MCTSInput(
            smiles=target,
            description=multi_target_input.description,
            tags=multi_target_input.tags,
            expand_one_options=multi_target_input.expand_one_options,
            build_tree_options=multi_target_input.build_tree_options,
            enumerate_paths_options=multi_target_input.enumerate_paths_options,
            run_async=multi_target_input.run_async,
            result_id=multi_target_input.result_id
        )

        return wrapper_input

    @staticmethod
    def get_paths_from_response(wrapper_response: MCTSResponse
                                ) -> list[dict[str, Any]] | None:
        result = wrapper_response.result
        if result is None:
            return None
        paths = result.paths

        return paths

    @staticmethod
    def convert_output_to_response(best_paths: dict[str, dict[str, Any] | None]
                                   ) -> MultiTargetResponse:
        status_code = 200
        message = "multi_target.call_sync_without_token() successfully executed."

        response = MultiTargetResponse(
            status_code=status_code,
            message=message,
            result=best_paths
        )

        return response
