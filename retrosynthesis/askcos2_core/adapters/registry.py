import importlib
import os
from adapters import ADAPTER_CLASSES

_adapter_registry = None


def get_adapter_registry():
    """Get global adapter registry."""
    global _adapter_registry
    if _adapter_registry is None:
        default_path = "configs.module_config_full"
        config_path = os.environ.get(
            "MODULE_CONFIG_PATH", default_path
        ).replace("/", ".").rstrip(".py")
        _adapter_registry = AdapterRegistry(config_path=config_path)
        print(f"Loaded adapter configuration from {config_path}")

    return _adapter_registry


class AdapterRegistry:
    def __init__(self, config_path: str):
        module_config = importlib.import_module(config_path).module_config
        self.module_config = module_config

        # Note that module_config is not really used here.
        # We'll turn all adapters on, so it suffices to iterate over ADAPTER_CLASSES
        self._adapters = {}
        for name, adapter_class in ADAPTER_CLASSES.items():
            self._adapters[name] = adapter_class()

    def get_adapter(self, module: str):
        return self._adapters.get(module, None)

    def __iter__(self):
        return iter(self._adapters.values())
