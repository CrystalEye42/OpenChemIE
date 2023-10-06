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
import re

BOND_TO_INT = {
    "": 0,
    "single": 1,
    "double": 2, 
    "triple": 3, 
    "aromatic": 4, 
    "solid wedge": 5, 
    "dashed wedge": 6
}

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

def process_tables(figures, results, molscribe, batch_size=16):
    r_group_pattern = re.compile(r'^(\d+-)?(?P<group>\w+)( \(\w+\))?$')
    for figure, result in zip(figures, results):
        result['page'] = figure['page']
        if figure['table']['content'] is not None:
            content = figure['table']['content']
            if len(result['reactions']) != 1:
                print("Warning: multiple reactions detected")
            orig_reaction = result['reactions'][0]
            graphs = get_atoms_and_bonds(figure['figure']['image'], orig_reaction, molscribe, batch_size=batch_size)
            relevant_locs = find_relevant_groups(graphs, content['columns'])
            for row in content['rows']:
                r_groups = {}
                expanded_conditions = orig_reaction['conditions'][:]
                for col, entry in zip(content['columns'], row):
                    if col['tag'] != 'alkyl group':
                        expanded_conditions.append({
                            'category': '[Table]',
                            'text': entry['text'], 
                            'tag': col['tag'],
                            'header': col['text'],
                        })
                    else:
                        found = r_group_pattern.match(entry['text'])
                        r_groups[col['text']] = found.group('group')
                reaction = get_replaced_reaction(orig_reaction, graphs, relevant_locs, r_groups, molscribe)  
                to_add ={
                    'reactants': reaction['reactants'][:],
                    'condition': expanded_conditions,
                    'products': reaction['products'][:]
                }
                result['reactions'].append(to_add)
    return results


def get_atoms_and_bonds(image, reaction, molscribe, batch_size=16):
    image = convert_to_cv2(image)
    cropped_images = []
    results = []
    for key, molecules in reaction.items():
        for i, elt in enumerate(molecules):
            if elt['category'] != '[Mol]':
                continue
            x1, y1, x2, y2 = elt['bbox']
            height, width, _ = image.shape
            cropped_images.append(image[int(y1*height):int(y2*height),int(x1*width):int(x2*width)])
            to_add = {
                'image': cropped_images[-1],
                'chartok_coords': {
                    'coords': [],
                    'symbols': [],
                },
                'edges': [],
                'key': (key, i)
            }
            results.append(to_add)
    outputs = molscribe.predict_images(cropped_images, return_atoms_bonds=True, batch_size=batch_size)
    for mol, result in zip(outputs, results):
        for atom in mol['atoms']:
            result['chartok_coords']['coords'].append((atom['x'], atom['y']))
            result['chartok_coords']['symbols'].append(atom['atom_symbol'])
        result['edges'] = [[0] * len(mol['atoms']) for _ in range(len(mol['atoms']))]
        for bond in mol['bonds']:
            i, j = bond['endpoint_atoms']
            result['edges'][i][j] = BOND_TO_INT[bond['bond_type']]
            result['edges'][j][i] = BOND_TO_INT[bond['bond_type']]
    return results

def find_relevant_groups(graphs, columns):
    results = {}
    r_groups = set([f"[{col['text']}]" for col in columns if col['tag'] == 'alkyl group'])
    for i, graph in enumerate(graphs):
        to_add = []
        for j, atom in enumerate(graph['chartok_coords']['symbols']):
            if atom in r_groups:
                to_add.append((atom[1:-1], j))
        results[i] = to_add
    return results

def get_replaced_reaction(orig_reaction, graphs, relevant_locs, mappings, molscribe):
    graph_copy = []
    for graph in graphs:
        graph_copy.append({
            'image': graph['image'],
            'chartok_coords': {
                'coords': graph['chartok_coords']['coords'][:],
                'symbols': graph['chartok_coords']['symbols'][:],
            },
            'edges': graph['edges'][:],
            'key': graph['key'],
        })
    for graph_idx, atoms in relevant_locs.items():
        for atom, atom_idx in atoms:
            if atom in mappings:
                graph_copy[graph_idx]['chartok_coords']['symbols'][atom_idx] = mappings[atom]
    reaction_copy = {}
    for k, v in orig_reaction.items():
        reaction_copy[k] = []
        for entity in v:
            if entity['category'] == '[Mol]':
                reaction_copy[k].append({
                    k1: v1 for k1, v1 in entity.items()
                })
            else:
                reaction_copy[k].append(entity)

    for graph in graph_copy:
        output = molscribe.convert_graph_to_output([graph], [graph['image']])
        molecule = reaction_copy[graph['key'][0]][graph['key'][1]]
        molecule['smiles'] = output[0]['smiles']
        molecule['molfile'] = output[0]['molfile']
    return reaction_copy

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
                        if len(Chem.GetMolFrags(tar_mol_removed, asMols = False)) == 1:
                            rdDepictor.Compute2DCoords(tar_mol_removed)
                            save = tar_mol_removed
                    site_end = get_sites(tar_mol, save, True)
                    start = sites_start
                    end = site_end[0]

                    mol1 = reactant_without_r
                    mol2 = save
                    mol3 = Chem.EditableMol(Chem.CombineMols(mol1, mol2))

                    Chem.EditableMol.AddBond(mol3, start, reactant_without_r.GetNumAtoms() + end, Chem.BondType.SINGLE)
                    toreturn[mol] = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol3.GetMol())))

        return toreturn
    else:
        pass


def associate_corefs(results, results_coref):
    coref_smiles = {}
    for result_coref in results_coref:
        bboxes, corefs = result_coref['bboxes'], result_coref['corefs']
        for coref in corefs:
            mol, idt = coref[0], coref[1]
            if len(bboxes[idt]['text']) > 0:
                coref_smiles[bboxes[idt]['text'][0]] = bboxes[mol]['smiles']
    for page in results:
        for reactions in page['reactions']:
            for reaction in reactions['reactions']:
                if 'Reactants' in reaction:
                    if isinstance(reaction['Reactants'], tuple):
                        if reaction['Reactants'][0] in coref_smiles:
                            reaction['Reactants'] = (f'{reaction["Reactants"][0]} ({coref_smiles[reaction["Reactants"][0]]})', reaction['Reactants'][1], reaction['Reactants'][2])
                    else:
                        for idx, compound in enumerate(reaction['Reactants']):
                            if compound[0] in coref_smiles:
                                reaction['Reactants'][idx] = (f'{compound[0]} ({coref_smiles[compound[0]]})', compound[1], compound[2])
                if 'Product' in reaction:
                    if isinstance(reaction['Product'], tuple):
                        if reaction['Product'][0] in coref_smiles:
                            reaction['Product'] = (f'{reaction["Product"][0]} ({coref_smiles[reaction["Product"][0]]})', reaction['Product'][1], reaction['Product'][2])
                    else:
                        for idx, compound in enumerate(reaction['Product']):
                            if compound[0] in coref_smiles:
                                reaction['Product'][idx] = (f'{compound[0]} ({coref_smiles[compound[0]]})', compound[1], compound[2])
    return results
