import torch
from functools import lru_cache
import layoutparser as lp
import pdf2image
from PIL import Image
from .reaction_model.predict_bbox import ReactionModel
from huggingface_hub import hf_hub_download
from molscribe import MolScribe
from rxnscribe import RxnScribe, MolDetect
from .utils import clean_bbox_output, get_figures_from_pages, convert_to_pil, convert_to_cv2

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
    
    @lru_cache(maxsize=None)
    def init_molscribe(self, ckpt_path=None):
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
                ckpt_path = hf_hub_download("Ozymandias314/MolDetectCkpt", "best.ckpt")
            else:
                ckpt_path = self.moldet_ckpt
        return MolDetect(ckpt_path)

    def extract_mol_info_from_pdf(self, pdf, batch_size=16, num_pages=None):
        """
        Get all molecules and their information from a pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {   # first figure
                    'image': ndarray of the figure image,
                    'molecules': [
                        {   # first molecule
                            'bbox': tuple in the form (x1, y1, x2, y2),
                            'score': float,
                            'image': ndarray of cropped molecule image,
                            'smiles': str,
                            'molfile': str
                        },
                        # more molecules
                    ],
                    'page': int
                },
                # more figures
            ]
        """
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages) 
        images = [figure['image'] for figure in figures]
        results = self.extract_mol_info_from_figures(images, batch_size=batch_size)
        for figure, result in zip(figures, results):
            result['page'] = figure['page']
        return results
    
    def extract_figures_from_pdf(self, pdf, num_pages=None):
        """
        Find and return all figures from a pdf
        Parameters:
            pdf: path to pdf, or byte file
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of figures in the following format
            [
                {   # first figure
                    'image': PIL image of figure,
                    'page': int
                },
                # more figures
            ]
        """
        pdfparser = self.init_pdfparser()
        pages = None
        if type(pdf) == str:
            pages = pdf2image.convert_from_path(pdf, last_page=num_pages)
        else:
            pages = pdf2image.convert_from_bytes(pdf, last_page=num_pages)
        
        return get_figures_from_pages(pages, pdfparser)

    def extract_mol_bboxes_from_figures(self, figures, batch_size=16):
        """
        Return bounding boxes of molecules in images
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference
        Returns:
            list of results for each figure in the following format
            [
                [   # first figure
                    {   # first bounding box
                        'category': str,
                        'bbox': tuple in the form (x1, y1, x2, y2),
                        'category_id': int,
                        'score': float
                    },
                    # more bounding boxes
                ],
                # more figures
            ]
        """
        figures = [convert_to_pil(figure) for figure in figures]
        moldet = self.init_moldet()
        return moldet.predict_images(figures, batch_size=batch_size)

    def extract_mol_info_from_figures(self, figures, batch_size=16):
        """
        Get all molecules and their information from list of figures
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference
        Returns:
            list of results for each figure in the following format
            [
                {   # first figure
                    'image': ndarray of the figure image,
                    'molecules': [
                        {   # first molecule
                            'bbox': tuple in the form (x1, y1, x2, y2),
                            'score': float,
                            'image': ndarray of cropped molecule image,
                            'smiles': str,
                            'molfile': str
                        },
                        # more molecules
                    ],
                },
                # more figures
            ]
        """
        bboxes = self.extract_mol_bboxes_from_figures(figures, batch_size=batch_size)
        figures = [convert_to_cv2(figure) for figure in figures]
        results, cropped_images, refs = clean_bbox_output(figures, bboxes)
        molscribe = self.init_molscribe()
        mol_info = molscribe.predict_images(cropped_images, batch_size=batch_size)
        for info, ref in zip(mol_info, refs):
            ref.update(info)
        return results
    
    def extract_rxn_info_from_pdf(self, pdf, batch_size=16, num_pages=None):
        """
        Get reaction information from figures in pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {
                    'figure': PIL image
                    'reactions': [
                        {
                            'reactants': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'smiles': str,
                                    'molfile': str,
                                },
                                # more reactants
                            ],
                            'conditions': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                },
                                # more conditions
                            ],
                            'products': [
                                # same structure as reactants
                            ]
                        },
                        # more reactions
                    ],
                    'page': int
                },
                # more figures
            ]
        """
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages) 
        images = [figure['image'] for figure in figures]
        results = self.extract_rxn_info_from_figures(images, batch_size=batch_size)
        for figure, result in zip(figures, results):
            result['page'] = figure['page']
        return results


    def extract_rxn_info_from_figures(self, figures, batch_size=16):
        """
        Get reaction information from list of figures
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference in all models
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {
                    'figure': PIL image
                    'reactions': [
                        {
                            'reactants': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'smiles': str,
                                    'molfile': str,
                                },
                                # more reactants
                            ],
                            'conditions': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                },
                                # more conditions
                            ],
                            'products': [
                                # same structure as reactants
                            ]
                        },
                        # more reactions
                    ],
                },
                # more figures
            ]

        """
        pil_figures = [convert_to_pil(figure) for figure in figures]
        rxnscribe = self.init_rxnscribe()
        results = []
        reactions = rxnscribe.predict_images(pil_figures, molscribe=True, ocr=False)
        for figure, rxn in zip(figures, reactions):
            data = {
                'figure': figure,
                'reactions': rxn,
                }
            results.append(data)
        return results
        


if __name__=="__main__":
    chemescribe = ChemEScribe()
