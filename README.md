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

## Usage
Importing all models:
```
import torch
from openchemie import OpenChemIE

model = OpenChemIE(device=torch.device('cpu')) # change to cuda for gpu
```
### List of Methods
 - [extract_molecules_from_figures_in_pdf](#extracting-molecule-information-from-pdfs)
 - [extract_molecules_from_text_in_pdf](#extracting-molecule-information-from-pdfs)
 - [extract_reactions_from_figures_in_pdf](#extracting-reaction-information-from-pdfs)
 - [extract_reactions_from_text_in_pdf](#extracting-reaction-information-from-pdfs)
 - [extract_molecule_corefs_from_figures_in_pdf](#extracting-molecule-corefs-from-pdfs)
 - [extract_molecules_from_figures](#extracting-molecules-reactions-bounding-boxes-and-corefs-from-images)
 - [extract_reactions_from_figures](#extracting-molecules-reactions-bounding-boxes-and-corefs-from-images)
 - [extract_molecule_bboxes_from_figures](#extracting-molecules-reactions-bounding-boxes-and-corefs-from-images)
 - [extract_molecule_corefs_from_figures](#extracting-molecules-reactions-bounding-boxes-and-corefs-from-images)
 - [extract_figures_from_pdf](#extracting-figures-and-tables-from-pdfs)
 - [extract_tables_from_pdf](#extracting-figures-and-tables-from-pdfs)

### Extracting Molecule Information From PDFs

```
import torch
from openchemie import OpenChemIE

model = OpenChemIE()
pdf_path = 'path/to/pdf'
figure_results = model.extract_molecules_from_figures_in_pdf(pdf_path)
text_results = model.extract_molecules_from_text_in_pdf(pdf_path)
```

The output when extracting molecules from figures has the following format
```
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
```

Output when extracting molecules from text has the following format
```
[
    {
        'sentences': [
          [str], # list of tokens
          # more sentences
        ],
        'predictions': [
          [str], # same lengths as corresponding lists in `sentences`
          # more sentences
        ],
        'page': int
    },
    # more pages
]
```

### Extracting Reaction Information From PDFs
```
import torch
from openchemie import OpenChemIE

model = OpenChemIE()
pdf_path = 'path/to/pdf'
figure_results = model.extract_reactions_from_figures_in_pdf(pdf_path)
text_results = model.extract_reactions_from_text_in_pdf(pdf_path)
```

The output when extracting reactions from figures has the following format
```
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
                        'text': list of str,
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
```
Output when extracting reactions from text has the following format
```
[
    {
        'page': int,
        'reactions': [
            {
                'tokens': list of words in relevant sentence,
                'reactions' : [
                    {
                        'Reactants': list of tuple,
                        'Products': list of tuple,
                    },
                    # more reactions
                ]
            },
            # reactions in other sentences
        ]
    },
    # more pages
]
```

### Extracting Molecule Corefs From PDFs
```
import torch
from openchemie import OpenChemIE

model = OpenChemIE()
pdf_path = 'path/to/pdf'
results = model.extract_molecule_corefs_from_figures_in_pdf(pdf_path)
```
The output has the following format
```
[
    {
        'bboxes': [
            {   # first bbox
                'category': '[Sup]', 
                'bbox': (0.0050025012506253125, 0.38273870663142223, 0.9934967483741871, 0.9450094869920168), 
                'category_id': 4, 
                'score': -0.07593922317028046
            },
            # More bounding boxes
        ],
        'coref': [
            [0, 1],
            [3, 4],
            # More coref pairs
        ],
        'page': int
    },
    # More figures
]
```

### Extracting Molecules, Reactions, Bounding Boxes, and Corefs From Images
```
import torch
from openchemie import OpenChemIE
import cv2
from PIL import Image

model = OpenChemIE()

img = cv2.imread('path/to/img')
img2 = cv2.imread('path/to/other_img')
img3 = Image.open('path/to/img3')
images = [img, img2, img3] # supports both cv2 and PIL images

molecule_results = model.extract_molecules_from_figures(images)
reaction_results = model.extract_reactions_from_figures(images)
bbox_results = model.extract_molecule_bboxes_from_figures(images)
coref_results = model.extract_molecule_corefs_from_figures(images)
```

The output format when extracting molecules from images
```
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
```

The output format when extracting reactions from images
```
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
                        'text': list of str,
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
```

The output format when extracting molecule bounding boxes from images
```
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
```

The output format when extracting molecule corefs from images
```
[
    {
        'bboxes': [
            {   # first bbox
                'category': '[Sup]', 
                'bbox': (0.0050025012506253125, 0.38273870663142223, 0.9934967483741871, 0.9450094869920168), 
                'category_id': 4, 
                'score': -0.07593922317028046
            },
            # More bounding boxes
        ],
        'coref': [
            [0, 1],
            [3, 4],
            # More coref pairs
        ],
    },
    # More figures
]
```

### Extracting Figures and Tables From PDFs
```
import torch
from openchemie import OpenChemIE

model = OpenChemIE()
pdf_path = 'path/to/pdf'
figures = model.extract_figures_from_pdf(pdf_path, output_bbox=False, output_image=True)
tables = model.extract_tables_from_pdf(pdf_path, output_bbox=False, output_image=True)
```

Output format when extracting figures
```
[
    {   # first figure
        'title': str,
        'figure': {
            'image': PIL image or None,
            'bbox': list in form [x1, y1, x2, y2],
        }
        'table': {
            'bbox': list in form [x1, y1, x2, y2] or empty list,
            'content': {
                'columns': list of column headers,
                'rows': list of list of row content,
            } or None
        }
        'footnote': str or empty,
        'page': int
    },
    # more figures
]
```

Output format when extracting tables
```
[
    { # first table
        'title': str,
        'figure': {
            'image': PIL image or None,
            'bbox': list in form [x1, y1, x2, y2] or empty list,
        }
        'table': {
            'bbox': list in form [x1, y1, x2, y2] or empty list,
            'content': {
                'columns': list of column headers,
                'rows': list of list of row content,
            }
        }
        'footnote': str or empty,
        'page': int
    },
    # more tables
]
```

### Loading custom model checkpoints
List of different model names
 - molscribe
 - rxnscribe
 - pdfparser
 - moldet
 - chemrxnextractor
 - chemner
 - coref

To load a specific checkpoint for a model, pass in your path to checkpoint to its init method. For example
```
import torch
from openchemie import OpenChemIE

model = OpenChemIE()
model.init_molscribe('/path/to/ckpt')
```
