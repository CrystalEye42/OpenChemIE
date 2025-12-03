import glob
import importlib
import os

TOOLTIPS: dict[str, dict] = {}


def import_tooltips():
    fl = glob.glob(os.path.join("tooltips", "*.py"))

    for file in fl:
        basename = os.path.basename(file)
        if (
            not basename.startswith("_")
            and not basename.startswith(".")
            and (basename.endswith(".py"))
        ):
            tooltip_module_path = file[:file.find(".py")]
            tooltip_module_path = tooltip_module_path.replace("/", ".")
            module = importlib.import_module(tooltip_module_path)
            content = getattr(module, "content")
            tooltip_category = tooltip_module_path.split(".")[-1]
            TOOLTIPS.update({tooltip_category: content})


# Automatically import contents in any Python files in the tooltips/ directory,
import_tooltips()
