import torch
import numpy as np
from base64 import encodebytes
from PIL import Image
import cv2
import io
import layoutparser as lp
import pdf2image
from reaction_model.predict_bbox import ReactionModel
from huggingface_hub import hf_hub_download
from molscribe import MolScribe
from rxnscribe import RxnScribe

class ChemEScribe:
    def __init__(self):
        self.models = Models()

    def predict_pdf(self, pdf, reaction=False, batch_size=16, num_pages=None):
        figures = self.models.pdf_to_figures(pdf, num_pages=num_pages) 
        images = [figure['unencoded_image'] for figure in figures]
        results = self.predict_figures(images, reaction=reaction, batch_size=batch_size)
        for data in results:
            print(data.keys())
            # clean the output
        return results
       
    def predict_figures(self, figures, reaction=False, batch_size=16):
        results = []
        if reaction:
            reactions = []
            for i in range(0, len(figures), batch_size):
                batch = figures[i: min(batch_size+i, len(figures))]
                reactions.extend(model.get_reaction_info(batch))
            for figure, rxn in zip(figures, reactions):
                data = {}
                data['reactions'] = rxn
                results.append(data)
        
        else:
            output_bboxes = []
            for i in range(0, len(figures), batch_size):
                batch = figures[i: min(batch_size+i, len(figures))]
                output_bboxes.extend(self.models.figures_to_molecules(batch))
            for i, output in enumerate(output_bboxes):
                mol_bboxes = [elt['bbox'] for elt in output if elt['category'] == '[Mol]']
                mol_scores = [elt['score'] for elt in output if elt['category'] == '[Mol]']
                unique_bboxes = []
                scores = []
                data = {}
                results.append(data)
                data['image'] = figures[i]
                data['mol_bboxes'] = unique_bboxes 
                data['mol_scores'] = scores
                for bbox, score in zip(mol_bboxes, mol_scores):
                    if is_unique_bbox(bbox, unique_bboxes):
                        unique_bboxes.append(bbox)
                        scores.append(score)
            
            image_buffer = []
            buffer_keys = []
            for data in results:
                data['smiles'] = []
                data['molblocks'] = []
                image = cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB) #need to change PIL to cv2
                for bbox in data['mol_bboxes']:
                    height, width, _ = image.shape
                    x1, y1, x2, y2 = bbox
                    cropped = image[int(y1*height):int(y2*height), int(x1*width):int(x2*width)]
                    image_buffer.append(cropped)
                    buffer_keys.append(data)
            smiles_results, mol_results = self.model.predict_images(image_buffer, batch_size=batch_size)
            for data, smiles, molblock in zip(buffer_keys, smiles_results, mol_results):
                data['smiles'].append(smiles)
                data['molblocks'].append(molblock)

        return results
    
    
    def predict_figure(self, figure, reaction=False):
        return self.predict_figures([figure], reaction=reaction)[0]

    def predict_images(self, images, batch_size=16):
        smiles = []
        molblocks = []
        for i in range(0, len(images), batch_size):
            batch = images[i: min(batch_size+i, len(images))]
            batch_smiles, batch_mols = self.models.get_molecule_info(batch)
            smiles.extend(batch_smiles)
            molblocks.extned(batch_mols)
        return smiles, molblocks

    def predict_image(self, image):
        smiles, molblocks = self.predict_images([image])
        return smiles[0], molblocks[0]


class Models:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layout_parser = None
        self.reaction = None
        self.molscribe = None
        self.rxnscribe = None

    def pdf_to_figures(self, pdf, num_pages=None):
        if self.layout_parser is None:
            self.layout = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config', 
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5], 
                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, 
                device=self.device)
        images = None
        if type(pdf) == str:
            images = pdf2image.convert_from_path(pdf, last_page=num_pages)
        else:
            images = pdf2image.convert_from_bytes(pdf, last_page=num_pages)
        
        figures = []
        count = 1
        for i in range(len(images)):
            img = np.asarray(images[i])
            layout = self.layout.detect(img)
            blocks = lp.Layout([b for b in layout if b.type in ["Figure", "Table"]])
            for block in blocks:
                image = Image.fromarray(block.crop_image(img))
                data = {'unencoded_image': image}

                # get encoded version of image to return
                byte_arr = io.BytesIO()
                image.save(byte_arr, format='PNG')
                encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
                data['image'] = encoded_img
                figures.append(data)
        return figures


    def figures_to_molecules(self, figures):
        if self.reaction is None:
            self.reaction = ReactionModel()
        return self.reaction.predict(figures)


    def get_molecule_info(self, images):
        if self.molscribe is None:
            ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
            self.molscribe = MolScribe(ckpt_path, device=torch.device(self.device))
        results = self.molscribe.predict_images(images)
        smiles_results = [r['smiles'] for r in results]
        molfile_results = [r['molfile'] for r in results]
        return smiles_results, molfile_results

    def get_reaction_info(self, figures):
        if self.rxnscribe is None:
            ckpt_path2 = hf_hub_download("yujieq/RxnScribe", "pix2seq_reaction_full.ckpt")
            self.rxnscribe = RxnScribe(ckpt_path2, device=torch.device(self.device))
        return self.rxnscribe.predict_images(figures, molscribe=True, ocr=False)



if __name__=="__main__":
    chemescribe = ChemEScribe()
    pdf_path = '/scratch/wang7776/chem_ie/ChemInfoExtractor/frontend/public/example1.pdf'
    print(chemescribe.predict_pdf(pdf_path))
    print(chemescribe.predict_pdf(pdf_path, reaction=True))
