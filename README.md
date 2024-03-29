# OpenChemIE
Authors: Yujie Qian, Alex Wang, Vincent Fan, Amber Wang, Regina Barzilay *(MIT CSAIL)*

OpenChemIE is an open source toolkit for aiding with chemistry information extraction by offering methods for extracting molecule or reaction data from figures or text. Given PDFs from chemistry literature, we run specialized machine learning models to efficiently extract structured data. For text analysis, we provide methods for named entity recognition and reaction extraction. For figure analysis, we offer methods for molecule detection, text-figure coreference, molecule recognition, and reaction diagram parsing. For more information on the models involved, [see below](#models-in-openchemie). 

## Citation

## Installation
First create and activate a [conda](https://numdifftools.readthedocs.io/en/stable/how-to/create_virtual_env_with_conda.html) virtual environment with the following
```
conda create -n openchemie python=3.9
conda activate openchemie
```
Run the following commands to install the package and its dependencies
```
conda install -c conda-forge pycocotools==2.0.4
pip install 'OpenChemIE @ git+https://github.com/CrystalEye42/OpenChemIE'
pip uninstall MolScribe
pip install --no-deps 'MolScribe @ git+https://github.com/CrystalEye42/MolScribe.git@cb08600'
```

Alternatively, for development of the package, clone and install as editable with the following
```
git clone https://github.com/CrystalEye42/OpenChemIE.git
cd OpenChemIE
pip install --editable .
```

Additionally, if Poppler is not already installed on your system, follow the corresponding [installation instructions](https://github.com/jalan/pdftotext#os-dependencies) for your OS.

## Usage
Importing all models:
```python
import torch
from openchemie import OpenChemIE

model = OpenChemIE(device=torch.device('cpu')) # change to cuda for gpu
```
### List of Methods
 - [extract_molecules_from_figures_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L245)
 - [extract_molecules_from_text_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L513)
 - [extract_reactions_from_figures_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L404)
 - [extract_reactions_from_text_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L560)
 - [extract_molecule_corefs_from_figures_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L336)
 - [extract_molecules_from_figures](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L303)
 - [extract_reactions_from_figures](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L457)
 - [extract_molecule_bboxes_from_figures](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L279)
 - [extract_molecule_corefs_from_figures](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L373)
 - [extract_figures_from_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L165)
 - [extract_tables_from_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L205)
 - [init methods for models](#loading-custom-model-checkpoints)

### Extracting Molecule Information From PDFs
These methods are for identifying and translating molecules in figures to their chemical structures, as well as for named entity recognition from texts. 
 - [extract_molecules_from_figures_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L245)
 - [extract_molecules_from_text_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L513)

```python
import torch
from openchemie import OpenChemIE

model = OpenChemIE()
pdf_path = 'example/acs.joc.2c00749.pdf'  # Change it to the path of your PDF
# Figure analysis
figure_results = model.extract_molecules_from_figures_in_pdf(pdf_path)
# Text analysis
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
        'molecules': [
            { # first paragraph
                'text': str,
                'labels': [
                    (str, int, int), # tuple of label, range start (inclusive), range end (exclusive)
                    # more labels
                ]
            },
            # more paragraphs
        ]
        'page': int
    },
    # more pages
]
```

### Extracting Reaction Information From PDFs
These methods are for parsing reaction schemes and conditions from figures or from text. 
 - [extract_reactions_from_figures_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L404)
 - [extract_reactions_from_text_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L560)

```python
import torch
from openchemie import OpenChemIE

model = OpenChemIE()
pdf_path = 'example/acs.joc.2c00749.pdf'
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
visualization method (utility function)
an example of the output

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
                        # key, value pairs where key is the label and value is a tuple of the form (tokens, start index, end index)
                        # where indices are for the corresponding token list and start and end are inclusive
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
This method is for resolving coreferences between molecules in the figure and labels in the text.
 - [extract_molecule_corefs_from_figures_in_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L336)

```python
import torch
from openchemie import OpenChemIE

model = OpenChemIE()
pdf_path = 'example/acs.joc.2c00749.pdf'
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
            ([0, 1], "4a"), #coreferences contain the indices of the molecule and identifier bounding boxes, as well as the string representation of the identifier
            ([3, 4], "4b"),
            # More coref pairs
        ],
        'page': int
    },
    # More figures
]
```

### Extracting From a List of Figures
The previous methods were for extracting directly from a PDF file. Below, we provide corresponding methods for extracting from a list of figure images instead. 

 - [extract_molecules_from_figures](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L303)
 - [extract_reactions_from_figures](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L457)
 - [extract_molecule_bboxes_from_figures](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L279)
 - [extract_molecule_corefs_from_figures](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L373)

```python
import torch
from openchemie import OpenChemIE
import cv2
from PIL import Image

model = OpenChemIE()

img = cv2.imread('example/img1.png')
img2 = cv2.imread('example/img2.png')
img3 = Image.open('example/img3.png')
images = [img, img2, img3] # supports both cv2 and PIL images

molecule_results = model.extract_molecules_from_figures(images)
reaction_results = model.extract_reactions_from_figures(images)
bbox_results = model.extract_molecule_bboxes_from_figures(images)
coref_results = model.extract_molecule_corefs_from_figures(images)
```

The output format for these methods are largely the same as their corresponding PDF methods, just missing the `'page'` key. However, for extracting molecule bounding boxes from images, (which doesn't have a corresponding method,) the output has the following format

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

### Extracting Figures and Tables From PDFs
These are helper methods for extracting just tables or figures without performing further analysis on them.
 - [extract_figures_from_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L165)
 - [extract_tables_from_pdf](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L205)

```python
import torch
from openchemie import OpenChemIE

model = OpenChemIE()
pdf_path = 'example/acs.joc.2c00749.pdf'
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

## Data
Data for evaluating the process of R group resolution is found at this [huggingface repository](https://huggingface.co/datasets/Ozymandias314/OpenChemIEData/tree/main). The huggingface repository contains every diagram in the dataset [here](https://huggingface.co/datasets/Ozymandias314/OpenChemIEData/blob/main/r_group_resolution_diagrams.zip) as well as groundtruth annotations [here]( https://huggingface.co/datasets/Ozymandias314/OpenChemIEData/blob/main/r_group_resolution_data.json).

The annotations take the following format: 
```python
[
  {
    "file_name": "acs.joc.2c00176 example 1.png",
    "reaction_template": {
      "reactants": [
        "*C(=O)NN=CC(F)(F)F",
        "N#CN"
      ],
      "products": [
        "*C(=O)N1NC(C(F)(F)F)N=C1N"
      ]
    },
    "detailed_reactions": {
      "a": {
        "reactants": [
          "O=C(NN=CC(F)(F)F)c1ccccc1",
          "N#CN"
        ],
        "products": [
          "NC1=NC(C(F)(F)F)NN1C(=O)c1ccccc1"
        ]
      },
      # more reactions
    }
  },
  # more diagrams
]
```

Additionally, jupyter notebooks used during the annotation process can be downloaded [here](https://huggingface.co/datasets/Ozymandias314/OpenChemIEData/blob/main/r_group_annotation_notebooks.zip). 

Diagrams and data used in the comparison against Reaxys can also be found in the same huggingface repository. 

### Loading Custom Model Checkpoints
[Init methods for models](https://github.com/CrystalEye42/OpenChemIE/blob/main/openchemie/interface.py#L35)

To load a specific checkpoint for a model, pass in your path to checkpoint to its corresponding init method. For example, to change the checkpoint of MolScribe
```python
import torch
from huggingface_hub import hf_hub_download
from openchemie import OpenChemIE

model = OpenChemIE()

ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m680k.pth")
model.init_molscribe(ckpt_path)
```

# Models In OpenChemIE
- MolScribe
  - An image-to-graph model for molecular structure recognition
  - Paper: https://pubs.acs.org/doi/10.1021/acs.jcim.2c01480
  - Code: https://github.com/thomas0809/MolScribe
  - Demo: https://huggingface.co/spaces/yujieq/MolScribe
- RxnScribe
  - An image-to-sequence generation model for reaction diagram parsing
  - Paper: https://pubs.acs.org/doi/10.1021/acs.jcim.3c00439
  - Code: https://github.com/thomas0809/rxnscribe
  - Demo: https://huggingface.co/spaces/yujieq/RxnScribe
- MolDet and MolCoref
  - An image-to-sequence generation model for identifying molecule bounding boxes, or for resolving coreferences between labels and molecules
  - Code: https://github.com/Ozymandias314/MolDetect/tree/main
- ChemNER
  - A sequence labeling model for identifying chemical entities in text
  - Code: https://github.com/Ozymandias314/ChemIENER
- ChemRxnExtractor
  - A sequence labeling model for parsing reactions from text
  - Paper: https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c00284
  - Code: https://github.com/jiangfeng1124/ChemRxnExtractor


