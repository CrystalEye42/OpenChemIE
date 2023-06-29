from PyPDF2 import PdfReader, PdfWriter
import pdfminer.high_level
import pdfminer.layout
from operator import itemgetter
from chemrxnextractor import RxnExtractor

class TextReactionExtractor(object):
    def __init__(self, pdf, pn):
        self.pdf_file = pdf
        self.pages = pn
        self.model_dir="cre_models_v0.1" # directory saving both prod and role models
        self.rxn_extractor = RxnExtractor(self.model_dir)
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
            write_file = open(self.text_file, "w+")
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
                                L.append(text[curind:i+1] + "\n")
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
                            L.append(text[curind:i] + ".\n")
                        else:
                            L.append(text[curind:i] + "\n")
                            
            write_file.writelines(L)
            write_file.close()
            
            reactions = self.get_reactions(page_number=current_page_num)
            ans.append(reactions)
            
            current_page_num += 1
        
        return ans
            
    def extract_limited_pages(self):
        ans = []
        current_page_num = 1
        for page_layout in pdfminer.high_level.extract_pages(self.pdf_file, maxpages=self.pages):
            write_file = open(self.text_file, "w+")
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
                                L.append(text[curind:i+1] + "\n")
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
                            L.append(text[curind:i] + ".\n")
                        else:
                            L.append(text[curind:i] + "\n")
                            
            write_file.writelines(L)
            write_file.close()
            
            reactions = self.get_reactions(page_number=current_page_num)
            ans.append(reactions)
            
            current_page_num += 1
        
        return ans
    
    def get_reactions(self, page_number=None):
        test_file = self.text_file
        with open(test_file, "r") as f:
            sents = f.read().splitlines()
        rxns = self.rxn_extractor.get_reactions(sents)
        
        ret = []
        for r in rxns:
            if len(r['reactions']) != 0: ret.append(r)
        ans = {}
        ans.update({'page' : page_number})
        ans.update({'reactions' : ret})
        return ans

"""
file_name = '/Users/Amber/Desktop/MIT/UROP/data/acs.joc.5b00099.pdf'
test = TextReactionExtractor(file_name, None)
print(test.extract_reactions_from_text())
"""
"""
[
    {'tokens':
        ['Reaction', 'of', 'diphenylacetylene', 'with', 'complex', '19A', 'led', 'to', 'only', 'cycloheptadienone', '23A', 'in', '30%', 'yield.'],
    'reactions':
        [
            {'Reactants':
                [('diphenylacetylene', 2, 2), ('19A', 5, 5)],
            'Product':
                ('23A', 10, 10),
            'Yield':
                [('30%', 12, 12)]
            }
        ]
    },
    {'tokens':
        ['Reaction', 'of', 'diphenylacetylene', 'with', '(phenylcyclopropyl)-carbene', 'complex', '19B,vcycloheptadienone', '25', 'was', 'produced', 'in', '53%', 'yield.'],
    'reactions':
        [
            {'Reactants':
                [('diphenylacetylene', 2, 2), ('(phenylcyclopropyl)-carbene complex', 4, 5), ('19B,vcycloheptadienone', 6, 6)],
            'Product':
                ('25', 7, 7),
            'Yield': [('53%', 11, 11)]
            }
        ]
    }
]
"""

