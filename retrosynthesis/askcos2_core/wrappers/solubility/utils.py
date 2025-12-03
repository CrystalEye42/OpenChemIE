"""
Utilities for processing solubility prediction results.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


def clean_up_value(value, sigfigs=3, lower_limit=1e-1, upper_limit=1000):
    """
    Round the given value to the specified number of significant figures.

    If lower_limit <= abs(value) <= upper_limit, return fixed notation.
    Otherwise, return exponential notation.

    Args:
        value (float): number to format
        sigfigs (int, optional): number of significant figures
        lower_limit (float, optional): lower threshold for exponential notation
        upper_limit (float, optional): upper threshold for exponential notation

    Returns:
        str: rounded value
    """
    if value is None:
        return value
    if (lower_limit and abs(value) < lower_limit) or (
        upper_limit and abs(value) > upper_limit
    ):
        return np.format_float_scientific(
            value, precision=sigfigs - 1
        )  # precision only counts digits after decimal
    else:
        return np.format_float_positional(
            value, precision=sigfigs, fractional=False
        )  # precision counts sig digits


def convert_logs_to_mgml(solute, solubility):
    """
    Converts solubility from logS units to mg/mL

    Args:
        solute: SMILES of solute
        solubility: solubility value in log10(mol/L)
    """
    mol = Chem.MolFromSmiles(solute)
    molwt = Descriptors.MolWt(mol)
    # Units: (mol/L) * (g/mol) * (1000 mg/g) / (1000 mL/L)
    return (10 ** float(solubility)) * molwt


def add_solubility_conversion(result):
    """
    Add solubility in mg/mL to the result object

    Modifies objects in place

    Args:
        result (dict): solubility result dictionary
    """
    fields_to_convert = {
        "logST (method1) [log10(mol/L)]": "ST (method1) [mg/mL]",
        "logST (method2) [log10(mol/L)]": "ST (method2) [mg/mL]",
        "logS298 [log10(mol/L)]": "S298 [mg/mL]",
    }
    for orig, conv in fields_to_convert.items():
        solute = result["Solute"]
        solubility = result[orig]
        if solubility is not None:
            converted = clean_up_value(
                convert_logs_to_mgml(solute, float(solubility)),
                sigfigs=3,
                lower_limit=0,
            )
        else:
            converted = None
        result[conv] = converted


def postprocess_solubility_results(results):
    """
    Postprocess list of solubility results by adding unit conversions

    Modifies list items in place and also returns list

    Args:
        results (list): list of solubility result dictionaries
    """
    for result in results:
        add_solubility_conversion(result)
    return results
