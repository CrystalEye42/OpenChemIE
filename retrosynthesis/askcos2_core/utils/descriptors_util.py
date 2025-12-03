import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

ELEMENTS = {
    "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53
}


def molecular_weight(smiles: str) -> float:
    """
    Calculate exact molecular weight for the given SMILES string

    Args:
        smiles: SMILES string for which to calculate molecular weight

    Returns:
         float: exact molecular weight
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        molwt = Descriptors.ExactMolWt(mol)

        return molwt
    except Exception:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return 9999.0
        mol.UpdatePropertyCache(strict=False)
        molwt = Descriptors.ExactMolWt(mol)

        return molwt


def rms_molecular_weight(smiles: str) -> float:
    """Calculates the root-mean-square molecular weight for a given SMILES string

    Args:
        smiles: SMILES string for which to calculate root mean squared molecular weight

    Returns:
        float: root mean squared molecular weight

    """
    smiles_split = smiles.split(".")
    molwt_list = [molecular_weight(smi) for smi in smiles_split]
    rms_molwt = np.sqrt(np.mean(np.square(molwt_list)))

    return float(rms_molwt)


def number_of_rings(smiles: str) -> int:
    """Calculates the number of rings in a given SMILES string

    Args:
        smiles: SMILES string for which to calculate the number of rings

    Returns:
        int: number of rings

    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 9999

    return int(mol.GetRingInfo().NumRings())

def number_of_heavy_atoms(smiles: str) -> int:
    """Calculates the number of heavy atoms in a given SMILES string
    
    Args:
        smiles: SMILES string for which to calculate the number of heavy atoms
    
    Returns:
        int: number of heavy atoms
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 9999
    return int(mol.GetNumHeavyAtoms())


def single_element_atom_count(mol: Chem.Mol, atomic_num: int) -> int:
    """Calculates the number of atoms for a single element in a given SMILES string
    
    Args:
        smiles: SMILES string for which to calculate the number of atoms with a given atomic number
        atomic_num: atomic number of the atoms to count
    
    Returns:
        int: number of atoms with a given atomic number
    """

    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_num)



def element_counts(smiles: str, elements_dict: dict = ELEMENTS) -> dict:
    """Calculates the number of atoms for each element in a given SMILES string
    
    Args:
        smiles: SMILES string for which to calculate the number of atoms with each atomic number
    
    Returns:
        dict: number of atoms with each atomic number
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {element: 9999 for element, _ in elements_dict.items()}
    return {
        element: 
        single_element_atom_count(mol, atomic_num) 
        for element, atomic_num in elements_dict.items()
    }


def get_core_fragment(smiles):
    """
    Extract the core fragment from a molecule for identifying near cycles in IPP (Inverse Planning Problem).
    
    The core fragment is defined as the union of:
        - All atoms that have atom mapping numbers (mapped atoms)
        - All carbon atoms that are directly connected to mapped carbon atoms
        - All ring atoms that are part of any ring containing the above atoms
    
    Non-core atoms are processed as follows:
        - Atoms connected to core atoms are replaced with dummy atoms (atomic number 0)
        - Completely disconnected atoms are removed from the molecule

    
    Args:
        smiles (str): SMILES string of the molecule to extract core fragment from.

    
    Returns:
        tuple: A tuple containing:
            - str: SMILES string of the core fragment with dummy atoms for peripheral groups,
                  or None if the input SMILES cannot be parsed
            - str: Canonical SMILES of the original molecule with all atom mappings removed,
                  or None if the input SMILES cannot be parsed
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    mol = Chem.RWMol(mol)

    mapped = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}

    
    keep_atoms = set(mapped)
    [x.SetAtomMapNum(0) for x in mol.GetAtoms()]
    
    canon_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    # Add ring atoms that are in rings with any current kept atoms
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if any(idx in keep_atoms for idx in ring):
            keep_atoms.update(ring)

    # Determine which atoms to replace with dummies or delete
    to_replace = set()
    to_delete = set()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in keep_atoms:
            continue
        neighbor_idxs = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        if any(n in keep_atoms for n in neighbor_idxs):
            to_replace.add(idx)
        else:
            to_delete.add(idx)

    # Replace atoms with dummies
    for idx in to_replace:
        atom = mol.GetAtomWithIdx(idx)
        atom.SetAtomicNum(0)
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetAtomMapNum(0)

    # Remove unconnected atoms
    for idx in sorted(to_delete, reverse=True):
        mol.RemoveAtom(idx)

    Chem.SanitizeMol(mol)

    # return larger fragment only
    frags = Chem.GetMolFrags(mol, asMols=True)
    largest_frag = max(frags, key=lambda m: m.GetNumAtoms())

    return Chem.MolToSmiles(largest_frag), canon_smiles

