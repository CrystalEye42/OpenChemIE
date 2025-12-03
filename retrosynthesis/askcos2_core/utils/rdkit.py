import json
import traceback as tb
from fastapi import Response
from pydantic import BaseModel
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDepictor, SDWriter, AllChem
from typing import Any
from utils import register_util
from utils.draw_impl import align_molecule
from utils.registry import get_util_registry
from io import StringIO
from utils.descriptors_util import (
    rms_molecular_weight, 
    molecular_weight,
    number_of_heavy_atoms, 
    number_of_rings, 
    element_counts, 
    get_core_fragment
)

# Re-export molecular_weight for external imports
__all__ = ['molecular_weight', 'RDKitUtil']

class SmilesInput(BaseModel):
    smiles: str
    isomericSmiles: bool = True
    reference: str = None


class MolfileInput(BaseModel):
    molfile: str
    isomericSmiles: bool = True


class RDKitAsyncReturn(BaseModel):
    request: dict
    task_id: str


def _canonicalize(_smi: str, isomericSmiles: bool) -> str:
    if _smi:
        _mol = Chem.MolFromSmiles(_smi)
        if not _mol:
            raise ValueError("Cannot parse smiles with rdkit.")
        _smi = Chem.MolToSmiles(_mol, isomericSmiles=isomericSmiles)
        if not _smi:
            raise ValueError("Cannot canonicalize smiles with rdkit.")
    return _smi

@register_util(name="rdkit")
class RDKitUtil:
    """Util class for RDKit"""
    prefixes = ["rdkit"]
    methods_to_bind: dict[str, list[str]] = {
        "canonicalize": ["POST"],
        "validate": ["POST"],
        "from_molfile": ["POST"],
        "to_molfile": ["POST"],
        "to_sdfile": ["POST"],
        "apply_one_template": ["POST"],
        "apply_one_template_by_idx": ["POST"],
        "get_rms_molecular_weight": ["POST"],
        "get_num_heavy_atoms": ["POST"],
        "get_num_rings": ["POST"],
        "get_element_counts": ["POST"],
        "get_core_fragment": ["POST"],
    }

    def __init__(self, util_config: dict[str, Any] = None):
        pass

    @staticmethod
    def canonicalize(input: SmilesInput) -> Response:
        smiles = input.smiles
        isomericSmiles = input.isomericSmiles

        resp = {}
        try:
            if ">" in smiles:
                resp["type"] = "rxn"
                resp["smiles"] = ">".join(
                    _canonicalize(part, isomericSmiles) for part in smiles.split(">")
                )
            else:
                resp["type"] = "mol"
                resp["smiles"] = _canonicalize(smiles, isomericSmiles)
        except ValueError:
            resp["error"] = f"Unable to canonicalize using RDKit, traceback: " \
                            f"{tb.format_exc()}"
            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )
        else:
            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )

    @staticmethod
    def validate(input: SmilesInput) -> Response:
        smiles = input.smiles

        mol = Chem.MolFromSmiles(smiles, sanitize=False)

        if mol is None:
            correct_syntax = False
            valid_chem_name = False
        else:
            correct_syntax = True
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                valid_chem_name = False
            else:
                valid_chem_name = True

        resp = {
            "correct_syntax": correct_syntax,
            "valid_chem_name": valid_chem_name,
        }

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )
  
    @staticmethod
    def from_molfile(input: MolfileInput):
        """
        Convert the provided Molfile to a SMILES string.

        Method: POST

        Parameters:

        - `molfile` (str): Molfile input
        - `isomericSmiles` (bool, optional): whether to generate isomeric SMILES

        Returns:

        - `smiles` (str): canonical SMILES
        """
        molfile = input.molfile
        isomericSmiles = input.isomericSmiles

        resp = {}

        mol = Chem.MolFromMolBlock(molfile)
        if not mol:
            resp["error"] = "Cannot parse sdf molfile with rdkit."

            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )

        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
        except Exception:
            resp["error"] = "Cannot parse sdf molfile with rdkit."

            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )

        resp["smiles"] = smiles

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )

    @staticmethod
    def to_molfile(input: SmilesInput):
        """
        Convert the provided SMILES string to a Molfile.

        Method: POST

        Parameters:

        - `smiles` (str): SMILES input
        - `reference` (str, optional): SMILES of reference molecule for alignment

        Returns:

        - `molfile` (str): Molfile output
        """

        smiles = input.smiles
        reference = input.reference

        resp = {}

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            resp["error"] = "Cannot parse smiles with rdkit."

            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )

        if reference:
            ref = Chem.MolFromSmiles(reference)
            align_molecule(mol, ref)
        else:
            rdDepictor.Compute2DCoords(mol)

        try:
            molfile = Chem.MolToMolBlock(mol)
        except Exception:
            resp["error"] = "Cannot parse smiles with rdkit."

            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )

        resp["molfile"] = molfile

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )

    @staticmethod
    def apply_one_template(smiles: str, reaction_smarts: str) -> list[str]:
        reaction_smarts_one = "(" + reaction_smarts.replace(">>", ")>>(") + ")"
        rxn = rdchiralReaction(str(reaction_smarts_one))
        prod = rdchiralReactants(smiles)

        try:
            reactants = rdchiralRun(rxn, prod, return_mapped=False)
        except:
            return []

        if not reactants:
            return []

        return reactants

    def apply_one_template_by_idx_sync(
        self,
        smiles: str,
        template_idx: int,
        template_set: str
    ) -> list[str]:
        template_controller = get_util_registry().get_util(module="template")
        reaction_smarts = template_controller.find_one_by_idx(
            template_idx=template_idx,
            template_set=template_set
        )["reaction_smarts"]

        try:
            return self.apply_one_template(
                smiles=smiles,
                reaction_smarts=reaction_smarts
            )
        except:
            return []

    @staticmethod
    async def apply_one_template_by_idx(
        smiles: str,
        template_idx: int,
        template_set: str,
        priority: int = 0
    ) -> RDKitAsyncReturn:
        from askcos2_celery.tasks import rdkit_apply_one_template_by_idx_task

        async_result = rdkit_apply_one_template_by_idx_task.apply_async(
            args=(smiles, template_idx, template_set), priority=priority)
        task_id = async_result.id

        request = {
            "smiles": smiles,
            "template_idx": template_idx,
            "template_set": template_set
        }
        async_return = RDKitAsyncReturn(
            request=request,
            task_id=task_id
        )

        return async_return
    
    @staticmethod
    async def to_sdfile(smiles: str)-> Response:
        resp = {}
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            resp["error"] = "Cannot parse smiles with rdkit."

            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )
        
        mol_with_hs = Chem.AddHs(mol)

        AllChem.EmbedMolecule(mol_with_hs)
        AllChem.UFFOptimizeMolecule(mol_with_hs)
    
        sdf_buffer = StringIO()
        writer = SDWriter(sdf_buffer)
        writer.write(mol_with_hs)
        writer.close()
    
        sdf_content = sdf_buffer.getvalue()
        sdf_buffer.close()

        resp["sdf"] = sdf_content

        return Response(
            content=json.dumps(resp),
            status_code=200,
            media_type="application/json"
        )
        
    @staticmethod
    def get_rms_molecular_weight(input: SmilesInput) -> Response:
        smiles = input.smiles

        resp = {}
        try:
            rms_molwt = rms_molecular_weight(smiles)
            resp["rms_molecular_weight"] = rms_molwt
        except Exception:
            resp["error"] = f"Unable to get RMS molecular weight using RDKit, traceback: " \
                            f"{tb.format_exc()}"
            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )
        else:
            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )

    @staticmethod
    def get_num_heavy_atoms(input: SmilesInput) -> Response:
        smiles = input.smiles

        resp = {}
        try:
            num_heavy_atoms = number_of_heavy_atoms(smiles)
            resp["num_heavy_atoms"] = num_heavy_atoms
        except Exception:
            resp["error"] = f"Unable to get number of heavy atoms using RDKit, traceback: " \
                            f"{tb.format_exc()}"
            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )
        else:
            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )

    @staticmethod
    def get_num_rings(input: SmilesInput) -> Response:
        smiles = input.smiles

        resp = {}
        try:
            num_rings = number_of_rings(smiles)
            resp["num_rings"] = num_rings
        except Exception:
            resp["error"] = f"Unable to get number of rings using RDKit, traceback: " \
                            f"{tb.format_exc()}"
            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )
        else:
            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )

    @staticmethod
    def get_element_counts(input: SmilesInput) -> Response:
        smiles = input.smiles

        resp = {}
        try:
            _element_counts = element_counts(smiles)
            resp["element_counts"] = _element_counts
        except Exception:
            resp["error"] = f"Unable to get element counts using RDKit, traceback: " \
                            f"{tb.format_exc()}"
            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )
        else:
            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )
    

    @staticmethod
    def get_core_fragment(input: SmilesInput) -> Response:
        smiles = input.smiles

        resp = {}
        try:
            core_fragment, canon_smiles = get_core_fragment(smiles)
            resp["core_fragment"] = core_fragment
            resp["canon_smiles"] = canon_smiles
        except Exception:
            resp["error"] = f"Unable to get core fragment using RDKit, traceback: " \
                            f"{tb.format_exc()}"
            return Response(
                content=json.dumps(resp),
                status_code=500,
                media_type="application/json"
            )
        else:
            return Response(
                content=json.dumps(resp),
                status_code=200,
                media_type="application/json"
            )
            


