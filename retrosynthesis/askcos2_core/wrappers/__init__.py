import glob
import importlib
import os
from pydantic import BaseModel

WRAPPER_CLASSES: dict[str, type] = {}


def register_wrapper(
    name: str,
    input_class: type[BaseModel],
    output_class: type[BaseModel],
    response_class: type[BaseModel]
):
    """
    New wrapper types can be added to askcos2_core with the
    :func:`register_wrapper` function decorator.

    For example::
        @register_wrapper(
            name='atom_map_indigo',
            input_class=AtomMapIndigoInput,
            output_class=AtomMapIndigoOutput,
            response_class=AtomMapIndigoResponse
        )
        class AtomMapIndigoWrapper(BaseWrapper):
            (...)

    Args:
        name (str): the name of the module
        input_class (class of type[BaseModel]): the input class of the module
        output_class (class of type[BaseModel]): the output class of the module
        response_class (class of type[BaseModel]): the response class of the module,
            which should have inherited from wrappers.base.BaseResponse
    """

    def register_wrapper_cls(cls):
        cls.name = name
        cls.input_class = input_class
        cls.output_class = output_class
        cls.response_class = response_class

        if name not in WRAPPER_CLASSES:
            WRAPPER_CLASSES[name] = cls

        return WRAPPER_CLASSES[name]

    return register_wrapper_cls


def import_wrappers():
    fl = glob.glob(os.path.join("wrappers", "*.py"))
    fl.extend(glob.glob(os.path.join("wrappers", "*", "*.py")))
    fl.extend(glob.glob(os.path.join("wrappers", "*", "*", "*.py")))

    for file in fl:
        basename = os.path.basename(file)
        if (
            not basename.startswith("_")
            and not basename.startswith(".")
            and (basename.endswith(".py"))
            and basename not in ["base.py", "registry.py"]
        ):
            wrapper_name = file[:file.find(".py")]
            wrapper_name = wrapper_name.replace("/", ".")
            importlib.import_module(wrapper_name)


# Automatically import any Python files in the wrappers/ directory,
# which will register all Wrapper classes with the decorator.
import_wrappers()
