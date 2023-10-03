import numpy as np
from PIL import Image
import cv2
import layoutparser as lp
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem

def get_figures_from_pages(pages, pdfparser):
    figures = []
    for i in range(len(pages)):
        img = np.asarray(pages[i])
        layout = pdfparser.detect(img)
        blocks = lp.Layout([b for b in layout if b.type == "Figure"])
        for block in blocks:
            figure = Image.fromarray(block.crop_image(img))
            figures.append({
                'image': figure,
                'page': i
            })
    return figures

def clean_bbox_output(figures, bboxes):
    results = []
    cropped = []
    references = []
    for i, output in enumerate(bboxes):
        mol_bboxes = [elt['bbox'] for elt in output if elt['category'] == '[Mol]']
        mol_scores = [elt['score'] for elt in output if elt['category'] == '[Mol]']
        data = {}
        results.append(data)
        data['image'] = figures[i]
        data['molecules'] = []
        for bbox, score in zip(mol_bboxes, mol_scores):
            x1, y1, x2, y2 = bbox
            height, width, _ = figures[i].shape
            cropped_img = figures[i][int(y1*height):int(y2*height),int(x1*width):int(x2*width)]
            cur_mol = {
                'bbox': bbox,
                'score': score,
                'image': cropped_img,
                #'info': None,
            }
            cropped.append(cropped_img)
            data['molecules'].append(cur_mol)
            references.append(cur_mol)
    return results, cropped, references
    
def convert_to_pil(image):
    if type(image) == np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    return image


def convert_to_cv2(image):
    if type(image) != np.ndarray:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return image

def get_sites(tar, ref, ref_site = False):
    rdDepictor.Compute2DCoords(ref)
    rdDepictor.Compute2DCoords(tar)
    idx_pair = rdDepictor.GenerateDepictionMatching2DStructure(tar, ref)

    in_template = [i[1] for i in idx_pair]
    sites = []
    for i in range(tar.GetNumAtoms()):
        if i not in in_template:
            for j in tar.GetAtomWithIdx(i).GetNeighbors():
                if j.GetIdx() in in_template and j.GetIdx() not in sites:

                    if ref_site: sites.append(idx_pair[in_template.index(j.GetIdx())][0])
                    else: sites.append(idx_pair[in_template.index(j.GetIdx())][0])
    return sites

def backout(results, coref_results):
    '''
    The inputs are raw outputs from extract_reactions_from_figures and extract_molecule_corefs_from_figures
    The output is a dictionary mapping products with an R group to their corresponding reactant. 
    Works when there is one reaction in the diagram with one R group, pending mistakes from OCR tool or whatever. 
    Possible improvements: generalize to multiple products, multiple R groups, additional smaller R groups in label. 
    '''
    reactants = [i['smiles'] for i in results[0]['reactions'][0]['reactants']]
    products = [i['smiles'] for i in results[0]['reactions'][0]['products']]
    coref_results_dict = {coref_results[0]['bboxes'][coref[0]]['smiles']: coref_results[0]['bboxes'][coref[1]]['text']  for coref in coref_results[0]['corefs']}
    
    if len(products) == 1:
        product_labels = coref_results_dict[products[0]]
        prod = products[0]
        if len(product_labels) == 1:
            
            idx = product_labels[0]
        else:
            pass
    else:
        pass
    
    idx_pattern = rf'{re.escape(idx)}[a-zA-Z]+'
    
    r_group_count = prod.count('*')
    if r_group_count == 1:
        prod_template = prod.replace('*', '')
        
        for reac in reactants:
            if '*' in reac:
                reactant = reac
                break
        reactant_template = reactant.replace('*', '')

        reactant_with_r = Chem.MolFromSmiles(reactant)

        reactant_without_r = Chem.MolFromSmiles(reactant_template)

        if reactant_without_r.GetNumAtoms() == 1:
            sites_start =0 
        else:
            sites_start = get_sites(reactant_with_r, reactant_without_r, True)[0]
        ref_mol_with_r_groups = Chem.MolFromSmiles(prod)
        ref_mol = Chem.MolFromSmiles(prod_template)
        toreturn = {}
        for mol in coref_results_dict:
            for parsed in coref_results_dict[mol]:
                if re.search(idx_pattern, parsed):
                    tar_mol = Chem.MolFromSmiles(mol)
                    substructs = tar_mol.GetSubstructMatches(ref_mol)
                    for sub in substructs:
                        editable = Chem.EditableMol(tar_mol)
                        for idx in tuple(sorted(sub, reverse = True)):
                            editable.RemoveAtom(idx)
                        tar_mol_removed = editable.GetMol()
                        print(Chem.GetMolFrags(tar_mol_removed, asMols = False))
                        if len(Chem.GetMolFrags(tar_mol_removed, asMols = False)) == 1:
                            print(Chem.GetMolFrags(tar_mol_removed, asMols = False))
                            rdDepictor.Compute2DCoords(tar_mol_removed)
                            save = tar_mol_removed
                    site_end = get_sites(tar_mol, save, True)
                    print(site_end)
                    start = sites_start
                    end = site_end[0]

                    mol1 = reactant_without_r
                    mol2 = save
                    mol3 = Chem.EditableMol(Chem.CombineMols(mol1, mol2))

                    Chem.EditableMol.AddBond(mol3, start, reactant_without_r.GetNumAtoms() + end, Chem.BondType.SINGLE)
                    print("wtf")
                    toreturn[mol] = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol3.GetMol())))

        return toreturn
        
        
    else:
        pass
