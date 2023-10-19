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
import copy

BOND_TO_INT = {
    "": 0,
    "single": 1,
    "double": 2, 
    "triple": 3, 
    "aromatic": 4, 
    "solid wedge": 5, 
    "dashed wedge": 6
}

RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'Rf', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar']

RGROUP_SYMBOLS = RGROUP_SYMBOLS + [f'[{i}]' for i in RGROUP_SYMBOLS]

RGROUP_SMILES = ['[1*]', '[2*]','[3*]', '[4*]','[5*]', '[6*]','[7*]', '[8*]','[9*]', '[10*]','[11*]', '[12*]','[a*]', '[b*]','[c*]', '[d*]','*', '[Rf]']

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
    r_group_pattern = re.compile(r'^(\w+-)?(?P<group>\w+)( \(\w+\))?$')
    for figure, result in zip(figures, results):
        result['page'] = figure['page']
        if figure['table']['content'] is not None:
            content = figure['table']['content']
            if len(result['reactions']) != 1:
                print("Warning: multiple reactions detected for table")
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
                        if found is not None:
                            r_groups[col['text']] = found.group('group')
                reaction = get_replaced_reaction(orig_reaction, graphs, relevant_locs, r_groups, molscribe)  
                to_add ={
                    'reactants': reaction['reactants'][:],
                    'conditions': expanded_conditions,
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

def get_atom_mapping(prod_mol, prod_smiles, prod = False, r_sites_reversed = None):
    # returns prod_mol_to_query which is the mapping of atom indices in prod_mol to the atom indices of the molecule represented by prod_smiles
    prod_template_intermediate = Chem.MolToSmiles(prod_mol)
    prod_template = prod_smiles
    for r in RGROUP_SMILES:
        if r!='*' and r!='(*)':
            prod_template = prod_template.replace(r, '*')
            prod_template_intermediate = prod_template_intermediate.replace(r, '*')
            
    prod_template_intermediate_mol = Chem.MolFromSmiles(prod_template_intermediate)
    prod_template_mol = Chem.MolFromSmiles(prod_template)
    
    p = Chem.AdjustQueryParameters.NoAdjustments()
    p.makeDummiesQueries = True
    
    prod_template_mol_query = Chem.AdjustQueryProperties(prod_template_mol, p)
    prod_template_intermediate_mol_query = Chem.AdjustQueryProperties(prod_template_intermediate_mol, p)
    rdDepictor.Compute2DCoords(prod_mol)
    rdDepictor.Compute2DCoords(prod_template_mol_query)
    rdDepictor.Compute2DCoords(prod_template_intermediate_mol_query)
    idx_pair = rdDepictor.GenerateDepictionMatching2DStructure(prod_mol, prod_template_intermediate_mol_query)
    
    intermdiate_to_prod_mol = {a:b for a,b in idx_pair}
    prod_mol_to_intermediate = {b:a for a,b in idx_pair}
    
    
    idx_pair_2 = rdDepictor.GenerateDepictionMatching2DStructure(prod_template_mol_query, prod_template_intermediate_mol_query)
    
    intermediate_to_query = {a:b for a,b in idx_pair_2}
    query_to_intermediate = {b:a for a,b in idx_pair_2}
    
    prod_mol_to_query = {a:intermediate_to_query[prod_mol_to_intermediate[a]] for a in prod_mol_to_intermediate}

    if prod:
        substructs = prod_template_mol_query.GetSubstructMatches(prod_template_intermediate_mol_query, uniquify = False)
    
        #idx_pair_2 = rdDepictor.GenerateDepictionMatching2DStructure(prod_template_mol_query, prod_template_intermediate_mol_query)
        for substruct in substructs:
            
            
            intermediate_to_query = {a:b for a, b in enumerate(substruct)}
            query_to_intermediate = {intermediate_to_query[i]: i for i in intermediate_to_query}

            prod_mol_to_query = {a:intermediate_to_query[prod_mol_to_intermediate[a]] for a in prod_mol_to_intermediate}
            
            good_map = True
            for i in r_sites_reversed:
                if prod_template_mol_query.GetAtomWithIdx(prod_mol_to_query[i]).GetSymbol() not in RGROUP_SMILES:
                    good_map = False
            if good_map:
                break

    return prod_mol_to_query, prod_template_mol_query



def backout(results, coref_results):
    reactants = results[0]['reactions'][0]['reactants']
    products = [i['smiles'] for i in results[0]['reactions'][0]['products']]
    coref_results_dict = {coref_results[0]['bboxes'][coref[0]]['smiles']: coref_results[0]['bboxes'][coref[1]]['text']  for coref in coref_results[0]['corefs']}
    
    
    if len(products) == 1:
        product_labels = coref_results_dict[products[0]]
        prod = products[0]
        if len(product_labels) == 1:
            # get the coreference label of the product molecule
            idx = product_labels[0]
        else:
            print("Warning: Malformed Label Parsed.")
            return
    else:
        print("Warning: More than one product detected")
        return
    
    # format the regular expression for labels that correspond to the product label
    numbers = re.findall(r'\d+', idx)
    idx = ''.join(numbers)
    idx_pattern = rf'{re.escape(idx)}[a-zA-Z]+'
    

    prod_smiles = prod
    prod_mol = Chem.MolFromMolBlock(results[0]['reactions'][0]['products'][0]['molfile'])
    
    # identify the atom indices of the R groups in the product tempalte
    r_sites = {}
    for idx, atom in enumerate(results[0]['reactions'][0]['products'][0]['atoms']):
        if atom['atom_symbol'] in RGROUP_SYMBOLS:
            r_sites[atom['atom_symbol']] = idx
    
    r_sites_reversed = {r_sites[i]: i for i in r_sites}
    
    num_r_groups = len(r_sites)

    #prepare the product template and get the associated mapping

    prod_mol_to_query, prod_template_mol_query = get_atom_mapping(prod_mol, prod_smiles, prod = True, r_sites_reversed = r_sites_reversed)
    
    reactant_mols = []
    
    toreturn = []
    #--------------process the reactants-----------------
    
    reactant_information = {} #index of relevant reaction --> [[R group name, atom index of R group, atom index of R group connection], ...]
    
    for idx, reactant in enumerate(reactants):
        reactant_information[idx] = []
        reactant_mols.append(Chem.MolFromSmiles(reactant['smiles']))
        has_r = False
        for a_idx, atom in enumerate(reactant['atoms']):
            
            #go through all atoms and check if they are an R group, if so add it to reactant information
            if atom['atom_symbol'] in r_sites:
                has_r = True
                reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                reactant_information[idx].append([atom['atom_symbol'], a_idx, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx).GetNeighbors()][0]])

        # if the reactant had r groups, we had to use the molecule generated from the MolBlock. 
        # but the molblock may have unexpanded elemeents that are not R groups
        # so we have to map back the r group indices in the molblock version to the full molecule generated by the smiles
        # and adjust the indices of the r groups accordingly
        if has_r:
            #get the mapping
            reactant_mol_to_query, _ = get_atom_mapping(reactant_mols[-1], reactant['smiles'])

            #make the adjustment
            for info in reactant_information[idx]:
                info[1] = reactant_mol_to_query[info[1]]
                info[2] = reactant_mol_to_query[info[2]]
            reactant_mols[-1] = Chem.MolFromSmiles(reactant['smiles'])

    #go through all the molecules in the coreference
    for other_prod in coref_results_dict:

        #check if they match the product label regex
        for parsed in coref_results_dict[other_prod]:
            if re.search(idx_pattern, parsed):

                other_prod_mol = Chem.MolFromSmiles(other_prod)

                # we get the substruct matches. note that we set uniquify to false since the order matters for our method
                substructs = other_prod_mol.GetSubstructMatches(prod_template_mol_query, uniquify = False)

                # for each substruct we create the mapping of the substruct onto the other_mol
                # delete all the molecules in other_mol correspond to the substruct
                # and check if they number of mol frags is equal to number of r groups
                # we do this to make sure we have the correct substruct
                for substruct in substructs:

                    query_to_other = {a:b for a,b in enumerate(substruct)}
                    other_to_query = {query_to_other[i]:i for i in query_to_other}

                    editable = Chem.EditableMol(other_prod_mol)
                    r_site_correspondence = []
                    for r in r_sites_reversed:
                        #get its id in substruct
                        substruct_id = query_to_other[prod_mol_to_query[r]]
                        r_site_correspondence.append([substruct_id, r_sites_reversed[r]])

                    for idx in tuple(sorted(substruct, reverse = True)):
                        if idx not in [query_to_other[prod_mol_to_query[i]] for i in r_sites_reversed]:
                            editable.RemoveAtom(idx)
                            for r_site in r_site_correspondence:
                                if idx < r_site[0]:
                                    r_site[0]-=1
                    other_prod_removed = editable.GetMol()
                    
                    if len(Chem.GetMolFrags(other_prod_removed, asMols = False)) == num_r_groups:
                        break
                
                # need to compute the sites at which correspond to each r_site_reversed

                r_site_correspondence.sort(key = lambda x: x[0])
                
                
                f = []
                ff = []
                frags = Chem.GetMolFrags(other_prod_removed, asMols = True, frags = f, fragsMolAtomMapping = ff)

                # r_group_information maps r group name --> the fragment/molcule corresponding to the r group and the atom index it should be connected at
                r_group_information = {}
                tosubtract = 0
                for idx, r_site in enumerate(r_site_correspondence):

                    r_group_information[r_site[1]]= (frags[f[r_site[0]]], r_site[0]-tosubtract)
                    tosubtract += len(ff[idx])
                    
                # now we modify all of the reactants according to the R groups we have found
                # for every reactant we disconnect its r group symbol, and connect it to the r group
                modify_reactants = copy.deepcopy(reactant_mols)
                modified_reactant_smiles = []
                for reactant_idx in reactant_information:
                    if len(reactant_information[reactant_idx]) == 0:
                        modified_reactant_smiles.append(Chem.MolToSmiles(modify_reactants[reactant_idx]))
                    else:
                        combined = reactant_mols[reactant_idx]
                        for r_group, r_index, connect_index in reactant_information[reactant_idx]:
                            combined = Chem.CombineMols(combined, r_group_information[r_group][0])

                        editable = Chem.EditableMol(combined)
                        atomIdxAdder = reactant_mols[reactant_idx].GetNumAtoms()
                        for r_group, r_index, connect_index in reactant_information[reactant_idx]:
                            Chem.EditableMol.RemoveBond(editable, r_index, connect_index)
                            Chem.EditableMol.AddBond(editable, connect_index, atomIdxAdder + r_group_information[r_group][1], Chem.BondType.SINGLE)
                            atomIdxAdder += r_group_information[r_group][0].GetNumAtoms()
                        r_indices = [i[1] for i in reactant_information[reactant_idx]]
                        
                        r_indices.sort(reverse = True)
                        

                        
                        for r_index in r_indices:
                            Chem.EditableMol.RemoveAtom(editable, r_index)
                            
                        modified_reactant_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(editable.GetMol()))))

                toreturn.append((modified_reactant_smiles, parsed))
    return toreturn


def associate_corefs(results, results_coref):
    coref_smiles = {}
    idx_pattern = r'\b\d+[a-zA-Z]{0,2}\b'
    for result_coref in results_coref:
        bboxes, corefs = result_coref['bboxes'], result_coref['corefs']
        for coref in corefs:
            mol, idt = coref[0], coref[1]
            if len(bboxes[idt]['text']) > 0:
                for text in bboxes[idt]['text']:
                    matches = re.findall(idx_pattern, text)
                    for match in matches:
                        coref_smiles[match] = bboxes[mol]['smiles']

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
