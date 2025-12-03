import json
from adapters.registry import get_adapter_registry
from celery import shared_task
from utils.registry import get_util_registry
from wrappers.registry import get_wrapper_registry


@shared_task
def base_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input).dict()

    return response


@shared_task
def legacy_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    adapter = get_adapter_registry().get_adapter(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = adapter.input_class(**input)
    response = adapter.call_sync(input).dict()

    return response


@shared_task
def context_recommender_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input).dict()

    return response


@shared_task
def count_analogs_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input).dict()

    return response


@shared_task
def forward_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input).dict()

    return response


@shared_task
def general_selectivity_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input).dict()

    return response


@shared_task
def impurity_predictor_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input).dict()

    return response


@shared_task
def pathway_ranker_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input).dict()

    return response


@shared_task
def retro_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input).dict()

    return response


@shared_task
def site_selectivity_task(module: str, input: dict) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input).dict()

    return response


# Tasks that require authentication (e.g., with the "token" arg)


@shared_task
def tree_analysis_task(module: str, input: dict, token: str) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input, token)
    json_response = json.loads(response.body.decode("utf-8"))
    json_response["status_code"] = response.status_code

    return json_response


@shared_task
def tree_search_expand_one_task(module: str, input: dict, token: str) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input, token).dict()

    return response


@shared_task
def tree_search_mcts_task(module: str, input: dict, token: str) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input, token).dict()

    return response


@shared_task
def tree_search_retro_star_task(module: str, input: dict, token: str) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input, token).dict()

    return response


@shared_task
def tree_search_task(module: str, input: dict, token: str) -> dict:
    """Celery tasks must have json serializable inputs/outputs"""
    wrapper = get_wrapper_registry().get_wrapper(module=module)

    # Reconstruct Input object from, and convert Output object to dict
    input = wrapper.input_class(**input)
    response = wrapper.call_sync(input, token).dict()

    return response


@shared_task
def rdkit_apply_one_template_by_idx_task(
    smiles: str,
    template_idx: int,
    template_set: str
) -> list[dict]:
    """Celery tasks must have json serializable inputs/outputs"""
    util = get_util_registry().get_util(module="rdkit")

    reactants = util.apply_one_template_by_idx_sync(
        smiles=smiles,
        template_idx=template_idx,
        template_set=template_set
    )

    response = [
        {
            "smiles": smiles,
            "template_idx": template_idx,
            "precursors": reactant.split("."),
            "ffscore": 0.99
        } for reactant in reactants
    ]

    return response
