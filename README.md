# OpenChemIE

This is a repository for aiding with chemistry information extraction by providing methods for easily using [RxnScribe](https://github.com/thomas0809/rxnscribe), [MolDetect](https://github.com/Ozymandias314/MolDetect), and [MolScribe](https://github.com/thomas0809/MolScribe) models. 

## Installation

Run the following command to install the package and its dependencies
```
python -m pip install 'OpenChemIE @ git+https://github.com/CrystalEye42/OpenChemIE'
```

## Quick Start
Example usage:
```
from openchemie import OpenChemIE
import cv2

model = OpenChemIE()

# Extracting molecules or reactions from a pdf
pdf_path = 'path/to/pdf'
mol_results = model.extract_molecules_from_pdf(pdf_path)
rxn_results = model.extract_reactions_from_pdf(pdf_path)

# Extracting from single image
img = cv2.imread('path/to/img')
mol_results = model.extract_molecules_from_figures([img]) 
rxn_results = model.extract_reactions_from_figures([img])

# Extracting from multiple images
img2 = cv2.imread('path/to/img2')
mol_results = model.extract_molecules_from_figures([img, img2]) 
rxn_results = model.extract_reactions_from_figures([img, img2])
```
