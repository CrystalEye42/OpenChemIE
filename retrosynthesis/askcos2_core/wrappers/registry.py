import importlib
import os
from pydantic import BaseModel
from wrappers import WRAPPER_CLASSES

_wrapper_registry = None


class BackendStatus(BaseModel):
    name: str
    description: str
    available_model_names: list[str]
    to_start: bool
    ready: bool
    backend_url: str | None


class BackendStatusResponse(BaseModel):
    modules: list[BackendStatus]


def get_wrapper_registry():
    """Get global wrapper registry."""
    global _wrapper_registry
    if _wrapper_registry is None:
        default_path = "configs.module_config_full"
        config_path = os.environ.get(
            "MODULE_CONFIG_PATH", default_path
        ).replace("/", ".").rstrip(".py")
        _wrapper_registry = WrapperRegistry(config_path=config_path)
        print(f"Loaded wrapper configuration from {config_path}")

    return _wrapper_registry


class WrapperRegistry:
    def __init__(self, config_path: str):
        module_config = importlib.import_module(config_path).module_config
        self.module_config = module_config

        self._wrappers = {}
        for module, to_start in module_config["modules_to_start"].items():
            if to_start:
                wrapper_config = module_config[module]

                for controller_prefix in [
                    "atom_map",
                    "forward",
                    "general_selectivity",
                    "retro",
                    "tree_search"
                ]:
                    controller_name = f"{controller_prefix}_controller"
                    if module.startswith(controller_prefix) and \
                            controller_name not in self._wrappers:
                        controller_class = WRAPPER_CLASSES[controller_name]
                        self._wrappers[controller_name] = controller_class()

                # Moved the wrapper binding behind controller so that the
                # controller doc appears first in the API doc
                if "wrapper_names" in wrapper_config:       # for multiple endpoints
                    for name in wrapper_config["wrapper_names"]:
                        wrapper_class = WRAPPER_CLASSES[name]
                        self._wrappers[name] = wrapper_class(wrapper_config)
                else:                                       # default to module name
                    wrapper_class = WRAPPER_CLASSES[module]
                    self._wrappers[module] = wrapper_class(wrapper_config)

        # tree analysis controller will be created always
        controller_class = WRAPPER_CLASSES["tree_analysis_controller"]
        self._wrappers["tree_analysis_controller"] = controller_class()

        # multi-target add-on, treating as a "controller" for now
        wrapper_class = WRAPPER_CLASSES["tree_search_multi_target"]
        self._wrappers["tree_search_multi_target"] = wrapper_class()

    def get_backend_status(self) -> BackendStatusResponse:
        """Endpoint for getting the status of backend services"""
        status_list = []

        for module, to_start in self.module_config["modules_to_start"].items():
            backend_url = ""
            ready = False

            if to_start:
                wrapper_config = self.module_config[module]
                if "wrapper_names" in wrapper_config:       # for multiple endpoints
                    wrapper = self.get_wrapper(
                        module=wrapper_config["wrapper_names"][0]
                    )
                else:
                    wrapper = self.get_wrapper(module=module)
                if wrapper is not None:
                    backend_url = wrapper.config["prediction_url_in_use"]
                    ready = wrapper.is_ready()

            backend_status = {
                "name": module,
                "description": self.module_config[module]["description"],
                "available_model_names":
                    self.module_config[module]["deployment"]["available_model_names"],
                "to_start": to_start,
                "ready": ready,
                "backend_url": backend_url
            }
            backend_status = BackendStatus(**backend_status)
            status_list.append(backend_status)

        resp = BackendStatusResponse(modules=status_list)

        return resp

    def get_wrapper(self, module: str):
        return self._wrappers.get(module, None)

    def __iter__(self):
        return iter(self._wrappers.values())
