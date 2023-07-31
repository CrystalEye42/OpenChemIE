from PyPDF2 import PdfReader, PdfWriter
import pdfminer.high_level
import pdfminer.layout
from operator import itemgetter
from chemrxnextractor import RxnExtractor

class TextReactionExtractor(object):
    def __init__(self, pdf, pn, device):
        self.pdf_file = pdf
        self.pages = pn
        self.model_dir="./training_modules/cre_models_v0.1" # directory saving both prod and role models
        use_cuda = (device == 'cuda')
        self.rxn_extractor = RxnExtractor(self.model_dir, use_cuda=use_cuda)
        self.text_file = "info.txt"
        
    def set_pdf_file(self, pdf):
        self.pdf_file = pdf
    
    def set_pages(self, pn):
        self.pages = pn
    
    def set_model_dir(self, md):
        self.model_dir = md
        self.rxn_extractor = RxnExtractor(self.model_dir)
    
    def set_text_file(self, tf):
        self.text_file = tf
    
    def extract_reactions_from_text(self):
        if self.pages == None:
            return self.extract_all_pages()
        else:
            return self.extract_limited_pages()
    
    def extract_all_pages(self):
        ans = []
        current_page_num = 1
        for page_layout in pdfminer.high_level.extract_pages(self.pdf_file):
            L = []
            for element in page_layout:
                if isinstance(element, pdfminer.layout.LTTextBoxHorizontal):
                    text = element.get_text()
                    text = text.replace("\n", " ")
                    text = text.replace("- ", "-")
                    curind = 0
                    i = 0
                    while i < len(text):
                        if text[i] == '.':
                            if i != 0 and not text[i-1].isdigit() or i != len(text) - 1 and (text[i+1] == " " or text[i+1] == "\n"):
                                L.append(text[curind:i+1])
                                while(i < len(text) and text[i] != " "):
                                    i += 1
                                curind = i + 1
                        i += 1
                    if curind != i:
                        if text[i - 1] == " ":
                            if i != 1:
                                i -= 1;
                            else:
                                break
                        if text[i - 1] != '.':
                            L.append(text[curind:i] + ".")
                        else:
                            L.append(text[curind:i])
            
            reactions = self.get_reactions(L, page_number=current_page_num)
            ans.append(reactions)
            
            current_page_num += 1
        
        return ans
            
    def extract_limited_pages(self):
        ans = []
        current_page_num = 1
        for page_layout in pdfminer.high_level.extract_pages(self.pdf_file, maxpages=self.pages):
            L = []
            for element in page_layout:
                if isinstance(element, pdfminer.layout.LTTextBoxHorizontal):
                    text = element.get_text()
                    text = text.replace("\n", " ")
                    text = text.replace("- ", "-")
                    curind = 0
                    i = 0
                    while i < len(text):
                        if text[i] == '.':
                            if i != 0 and not text[i-1].isdigit() or i != len(text) - 1 and (text[i+1] == " " or text[i+1] == "\n"):
                                L.append(text[curind:i+1])
                                while(i < len(text) and text[i] != " "):
                                    i += 1
                                curind = i + 1
                        i += 1
                    if curind != i:
                        if text[i - 1] == " ":
                            if i != 1:
                                i -= 1;
                            else:
                                break
                        if text[i - 1] != '.':
                            L.append(text[curind:i] + ".")
                        else:
                            L.append(text[curind:i])
                            
            reactions = self.get_reactions(L, page_number=current_page_num)
            ans.append(reactions)
            
            current_page_num += 1
        
        return ans
    
    def get_reactions(self, sents, page_number=None):
        rxns = self.rxn_extractor.get_reactions(sents)
        
        ret = []
        for r in rxns:
            if len(r['reactions']) != 0: ret.append(r)
        ans = {}
        ans.update({'page' : page_number})
        ans.update({'reactions' : ret})
        return ans

