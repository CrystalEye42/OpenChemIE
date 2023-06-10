import numpy as np
from PIL import Image
import cv2
import layoutparser as lp

def get_figures_from_pages(pages, pdfparser):
    figures = []
    for i in range(len(pages)):
        img = np.asarray(pages[i])
        layout = pdfparser.detect(img)
        blocks = lp.Layout([b for b in layout if b.type in ["Figure", "Table"]])
        for block in blocks:
            figure = Image.fromarray(block.crop_image(img))
            figures.append({
                'image': figure,
                'page': i
            })
    return figures

def get_overlap(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    u1, v1, u2, v2 = bbox2
    intersection_area = (min(u2,x2)-max(u1,x1))*(min(v2,y2)-max(v1,y1))
    return intersection_area/((x2-x1)*(y2-y1)+(u2-u1)*(v2-v1)-intersection_area)

def is_unique_bbox(bbox, bboxes):
    for b in bboxes:
        if get_overlap(b, bbox) > 0.9:
            return False
    return True

def clean_bbox_output(figures, bboxes):
    results = []
    cropped = []
    references = []
    for i, output in enumerate(bboxes):
        mol_bboxes = [elt['bbox'] for elt in output if elt['category'] == '[Mol]']
        mol_scores = [elt['score'] for elt in output if elt['category'] == '[Mol]']
        unique_bboxes = []
        data = {}
        results.append(data)
        data['image'] = figures[i]
        data['molecules'] = []
        for bbox, score in zip(mol_bboxes, mol_scores):
            if is_unique_bbox(bbox, unique_bboxes):
                unique_bboxes.append(bbox)
                x1, y1, x2, y2 = bbox
                height, width, _ = figures[i].shape
                cropped_img = figures[i][int(y1*height):int(y2*height),int(x1*width):int(x2*width)]
                cur_mol = {
                    'bbox': bbox,
                    'score': score,
                    #'image': cropped_img,
                    'info': None,
                }
                cropped.append(cropped_img)
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
