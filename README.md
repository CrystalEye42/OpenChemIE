# OpenChemIE

This is a repository for aiding with chemistry information extraction by providing methods for easily using the [RxnScribe](https://github.com/thomas0809/rxnscribe), [MolDetect](https://github.com/Ozymandias314/MolDetect), [MolScribe](https://github.com/thomas0809/MolScribe), [ChemRxnExtractor](https://github.com/jiangfeng1124/ChemRxnExtractor), and [ChemNER](https://github.com/Ozymandias314/ChemIENER) models. 

## Installation
(Optional but recommended.) First create and activate a virtual environment, such as by using [conda](https://numdifftools.readthedocs.io/en/stable/how-to/create_virtual_env_with_conda.html) or [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).


Run the following command to install the package and its dependencies
```
python -m pip install 'OpenChemIE @ git+https://github.com/CrystalEye42/OpenChemIE'
```

Alternatively, install from cloned repository with the following commands
```
git clone https://github.com/CrystalEye42/OpenChemIE.git
cd OpenChemIE
python setup.py install
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
