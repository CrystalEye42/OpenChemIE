import numpy as np
from PIL import Image
import cv2
import layoutparser as lp

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

