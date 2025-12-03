import glob
import importlib
import os
from pydantic import BaseModel

ADAPTER_CLASSES: dict[str, type] = {}


def register_adapter(
    name: str,
    input_class: type[BaseModel],
    result_class: type[BaseModel]
):
    """
    New adapter types can be added to askcos2_core with the
    :func:`register_adapter` function decorator.

    For example::
        @register_adapter(
            name='v1_selectivity',
            input_class=V1SelectivityInput
        )
        class V1SelectivityAdapter(BaseAdapter):
            (...)

    Args:
        name (str): the name of the module
        input_class (class of type[BaseModel]): the input class of the module
        result_class (class of type[BaseModel]): the result class of the module
    """

    def register_adapter_cls(cls):
        cls.name = name
        cls.input_class = input_class
        cls.result_class = result_class

        if name not in ADAPTER_CLASSES:
            ADAPTER_CLASSES[name] = cls

        return ADAPTER_CLASSES[name]

    return register_adapter_cls


def import_adapters():
    fl = glob.glob(os.path.join("adapters", "*.py"))
    fl.extend(glob.glob(os.path.join("adapters", "*", "*.py")))
    fl.extend(glob.glob(os.path.join("adapters", "*", "*", "*.py")))

    for file in fl:
        basename = os.path.basename(file)
        if (
            not basename.startswith("_")
            and not basename.startswith(".")
            and (basename.endswith(".py"))
            and basename not in ["base.py", "registry.py"]
        ):
            adapter_name = file[:file.find(".py")]
            adapter_name = adapter_name.replace("/", ".")
            importlib.import_module(adapter_name)


# Automatically import any Python files in the adapters/ directory,
# which will register all Adapter classes with the decorator.
import_adapters()
