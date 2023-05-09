import torch
from functools import lru_cache
import layoutparser as lp
import pdf2image
from .reaction_model.predict_bbox import ReactionModel
from huggingface_hub import hf_hub_download
from molscribe import MolScribe
from rxnscribe import RxnScribe
from .utils import clean_bbox_output, get_figures_from_pages

class ChemEScribe:
    def __init__(self, device=None, molscribe_ckpt=None, rxnscribe_ckpt=None, 
                 pdfparser_ckpt=None, moldet_ckpt=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.molscribe_ckpt = molscribe_ckpt
        self.rxnscribe_ckpt = rxnscribe_ckpt
        self.pdfparser_ckpt = pdfparser_ckpt
        self.moldet_ckpt = moldet_ckpt
    
    # @property
    # def molscribe(self):
    #     if self._molscribe is None:
    #         self._molscribe = self.init_molscribe()
    #     return self._molscribe

    # @molscribe.setter
    @lru_cache(maxsize=None)
    def init_molscribe(self, ckpt_path=None):
        print("initializing molscribe")
        if ckpt_path is None:
            if self.molscribe_ckpt is None:
                ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
            else:
                ckpt_path = self.molscribe_ckpt
        return MolScribe(ckpt_path, device=torch.device(self.device))
    
    @lru_cache(maxsize=None)
    def init_rxnscribe(self, ckpt_path=None):
        if ckpt_path is None:
            if self.rxnscribe_ckpt is None:
                ckpt_path = hf_hub_download("yujieq/RxnScribe", "pix2seq_reaction_full.ckpt")
            else:
                ckpt_path = self.rxnscribe_ckpt
        return RxnScribe(ckpt_path, device=torch.device(self.device))
    
    @lru_cache(maxsize=None)
    def init_pdfparser(self, ckpt_path=None):
        if ckpt_path is None:
            if self.pdfparser_ckpt is None:
                ckpt_path = "lp://efficientdet/PubLayNet/tf_efficientdet_d1"
            else:
                ckpt_path = self.pdfparser_ckpt
        return lp.AutoLayoutModel(ckpt_path, device=self.device)
    
    @lru_cache(maxsize=None)
    def init_moldet(self, ckpt_path=None):
        if ckpt_path is None:
            if self.moldet_ckpt is None:
                ckpt_path = hf_hub_download("wang7776/reaction_pix2seq", "best.ckpt")
            else:
                ckpt_path = self.moldet_ckpt
        return ReactionModel(ckpt_path, device=torch.device(self.device))

    def extract_mol_info_from_pdf(self, pdf, batch_size=16, num_pages=None):
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages) 
        images = [figure['image'] for figure in figures]
        results = self.extract_mol_info_from_figures(images, batch_size=batch_size)
        for data in results:
            print(data.keys())
            # clean the output
        return results
    
    def extract_figures_from_pdf(self, pdf, num_pages):
        pdfparser = self.init_pdfparser()
        pages = None
        if type(pdf) == str:
            pages = pdf2image.convert_from_path(pdf, last_page=num_pages)
        else:
            pages = pdf2image.convert_from_bytes(pdf, last_page=num_pages)
        
        return get_figures_from_pages(pages, pdfparser)

    def extract_mol_bboxes_from_figures(self, figures, batch_size=16):
        moldet = self.init_moldet()
        return moldet.predict(figures, batch_size=batch_size)

    def extract_mol_info_from_figures(self, figures, batch_size=16):
        bboxes = self.extract_mol_bboxes_from_figures(figures, batch_size=batch_size)
        results, cropped_images = clean_bbox_output(figures, bboxes)
        molscribe = self.init_molscribe()
        mol_info = molscribe.predict_images(cropped_images, batch_size=batch_size)
        # combine results
        return results
    
    def extract_rxn_info_from_pdf(self, pdf, batch_size=16, num_pages=None):
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages) 


    def extract_rxn_info_from_figures(self, figures, batch_size=16):
        rxnscribe = self.init_rxnscribe()
        results = []
        reactions = rxnscribe.predict_images(figures, molscribe=True, ocr=False)
        for figure, rxn in zip(figures, reactions):
            data = {
                'figure': figure,
                'reactions': rxn,
            }
            results.append(data)
        return results
        


if __name__=="__main__":
    chemescribe = ChemEScribe()
    pdf_path = '/scratch/wang7776/chem_ie/ChemInfoExtractor/frontend/public/example1.pdf'
    print(chemescribe.predict_pdf(pdf_path))
    print(chemescribe.predict_pdf(pdf_path, reaction=True))
