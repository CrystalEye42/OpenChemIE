"""
Module for generating annotated molecule images (SVG only)
"""
import math
import svgutils
from utils.draw_impl import molecule_smiles_to_image

gray = "#DDDDDD"
blue = "#007bff"  # Bootstrap blue
red = "#dc3545"  # Bootstrap red
LABEL_HEIGHT = 24
LABEL_MIN_WIDTH = 72


def element(name, **kwargs):
    """
    Convenience function for generating an svg element

    Args:
        name (str): element type to create
        **kwargs: element attributes to include

    Returns:
        str: XML string for the element
    """
    svg = f"<{name} "
    for key, value in kwargs.items():
        svg += f'{key.replace("_", "-")}="{value}" '
    svg += "/>\n"
    return svg


def text(contents, **kwargs):
    """
    Convenience function for generating an svg text element

    Args:
        contents (str): text contents
        **kwargs: element attributes to include

    Returns:
        str: XML string for the element
    """
    svg = "<text "
    for key, value in kwargs.items():
        svg += f'{key.replace("_", "-")}="{value}" '
    svg += ">"
    svg += contents
    svg += "</text>\n"
    return svg


def generate_annotation(width, ppg=0, as_reactant=0, as_product=0):
    """
    Generate annotation label for a molecule image.

    Args:
        width (float): width of molecule image in px
        ppg (float, optional): price of molecule
        as_reactant (int, optional): molecule popularity in training data as a reactant
        as_product (int, optional): molecule popularity in training data as a product

    Returns:
        str: standalone SVG of annotation as XML string
    """
    width = max(width, LABEL_MIN_WIDTH)

    height = LABEL_HEIGHT  # overall height of annotation bar
    midline_y = height / 2  # y coordinate of midline for centering elements
    font_size = 14
    stroke_width = 3
    circle_radius = 9

    def blue_circle(x, y, r):
        return element(
            "circle",
            cx=x,
            cy=y,
            r=r,
            fill="none",
            stroke=blue,
            stroke_width=stroke_width,
        )

    def red_circle_slash(x, y, r):
        # Distance from center to edge at 45deg, i.e. r * sin(45)
        offset = math.floor(r / math.sqrt(2))
        return element(
            "circle",
            cx=x,
            cy=y,
            r=r,
            fill="none",
            stroke=red,
            stroke_width=stroke_width,
        ) + element(
            "line",
            x1=x - offset,
            y1=y - offset,
            x2=x + offset,
            y2=y + offset,
            fill="none",
            stroke=red,
            stroke_width=2,
        )

    def centered_text(x, y, contents):
        return text(
            contents,
            x=x,
            y=y,
            font_size=font_size,
            font_family="Helvetica,Arial,sans-serif",
            text_anchor="middle",
            dominant_baseline="central",
        )

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" ' \
          f'width="{width}" height="{height}" version="1.1">\n'
    svg += element("rect", width=width, height=height, fill=gray)

    annotations = {
        "$": ppg > 0,
        "R": as_reactant > 0,
        "P": as_product > 0,
    }

    for i, (label, criteria) in enumerate(annotations.items()):
        x_pos = height * i + midline_y  # Space out equally based on height
        svg += centered_text(x_pos, midline_y, label)
        if criteria:
            svg += blue_circle(x_pos, midline_y, circle_radius)
        else:
            svg += red_circle_slash(x_pos, midline_y, circle_radius)

    svg += "</svg>"
    return svg


def generate_annotated_image(
    smiles, ppg=0, as_reactant=0, as_product=0, size=None, **kwargs
):
    """
    Generate an annotated molecule image.

    The size parameter determines scaling of the molecule image only and
    not the annotation. In addition, the annotation has a minimum width of 72px,
    so requesting sizes below 72px will not further reduce the image size.

    Args:
        smiles (str): SMILES of molecule to draw
        ppg (float, optional): price of molecule
        as_reactant (int, optional): molecule popularity in training data as a reactant
        as_product (int, optional): molecule popularity in training data as a product
        size (float, optional): length in pixels of shortest dimension in final image
        **kwargs: passed to ``molecule_smiles_to_image``

    Returns:
        str: standalone SVG of annotation as XML string
    """
    mol_svg = svgutils.transform.fromstring(molecule_smiles_to_image(smiles, **kwargs))
    mol_elem = mol_svg.getroot()

    mol_width = float(mol_svg.width.strip("px"))
    mol_height = float(mol_svg.height.strip("px"))

    # Calculate scaling for molecule if needed
    if size is not None:
        # Estimate final dimensions
        est_width = max(mol_width, LABEL_MIN_WIDTH)
        est_height = mol_height + LABEL_HEIGHT
        # Scale based on smallest dimension, accounting for label size
        if est_width > est_height:
            scale = min(1, (size - LABEL_HEIGHT) / mol_height)
        else:
            scale = min(1, max(size, LABEL_MIN_WIDTH) / mol_width)
        mol_height = math.ceil(mol_height * scale)
        mol_width = math.ceil(mol_width * scale)
        # Actual dimensions of space for molecule depends on requested size
        width = max(size, mol_width)
        height = max(size - LABEL_HEIGHT, mol_height)
    else:
        scale = 1
        width = mol_width
        height = mol_height

    label_svg = svgutils.transform.fromstring(
        generate_annotation(width, ppg, as_reactant, as_product)
    )
    label_elem = label_svg.getroot()

    label_width = float(label_svg.width.strip("px"))
    label_height = float(label_svg.height.strip("px"))

    # Translate label to below molecule
    label_elem.moveto(0, height)

    # Center molecule if necessary
    if mol_width < label_width or mol_height < height or scale != 1:
        x_offset = math.ceil((label_width - mol_width) / 2)
        y_offset = math.ceil((height - mol_height) / 2)
        mol_elem.moveto(x_offset, y_offset, scale_x=scale)

    overall_width = label_width
    overall_height = height + label_height

    fig = svgutils.compose.Figure(overall_width, overall_height, mol_elem, label_elem)

    return fig.tostr().decode("utf-8")
