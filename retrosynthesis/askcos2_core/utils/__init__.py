import glob
import importlib
import os

UTIL_CLASSES: dict[str, type] = {}


def register_util(name: str):
    """
    New util types can be added to askcos2_core with the
    :func:`register_util` function decorator.

    For example::
        @register_util(name='pricer')
        class Pricer:
            (...)

    Args:
        name (str): the name of the util module
    """

    def register_util_cls(cls):
        cls.name = name

        if name not in UTIL_CLASSES:
            UTIL_CLASSES[name] = cls

        return UTIL_CLASSES[name]

    return register_util_cls


def import_utils():
    fl = glob.glob(os.path.join("utils", "*.py"))
    fl.extend(glob.glob(os.path.join("utils", "*", "*.py")))
    fl.extend(glob.glob(os.path.join("utils", "*", "*", "*.py")))

    for file in fl:
        basename = os.path.basename(file)
        if (
            not basename.startswith("_")
            and not basename.startswith(".")
            and (basename.endswith(".py"))
            and basename not in ["base.py", "registry.py"]
        ):
            util_name = file[:file.find(".py")]
            util_name = util_name.replace("/", ".")
            importlib.import_module(util_name)


# Automatically import any Python files in the utils/ directory,
# which will register all Util classes with the decorator.
import_utils()
