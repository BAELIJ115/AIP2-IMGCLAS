# data_loader.py
# Loads ASCII image datasets and pairs each image with its label.

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Sample:
    """
    Represents one ASCII image and its label.
    - pixels: list of strings (each string is one row of the image)
    - label:  int (digit 0–9 or face/not-face indicator)
    """
    pixels: List[str]
    label: int


def load_ascii_dataset(image_path: str, label_path: str) -> Tuple[List[Sample], int, int]:
    """
    Loads an ASCII image dataset.

    Returns:
        samples : list of Sample objects
        height  : number of rows per image
        width   : number of columns per image
    """

    # ----------------------------
    # 1. Load labels
    # ----------------------------
    labels: List[int] = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line != "":
                labels.append(int(line))

    num_images = len(labels)
    if num_images == 0:
        raise ValueError("Label file contains no labels.")

    # ----------------------------
    # 2. Load image lines
    # ----------------------------
    image_lines: List[str] = []
    with open(image_path, "r") as f:
        for line in f:
            image_lines.append(line.rstrip("\n"))   # keep spaces, remove newline only

    # Remove trailing blank lines if present
    while image_lines and image_lines[-1] == "":
        image_lines.pop()

    total_lines = len(image_lines)

    if total_lines % num_images != 0:
        raise ValueError(
            f"Image file has {total_lines} lines but there are "
            f"{num_images} labels — cannot evenly divide into images."
        )

    # ----------------------------
    # 3. Infer dimensions
    # ----------------------------
    height = total_lines // num_images
    width = len(image_lines[0]) if height > 0 else 0

    # Validate that each line is the same width
    for i, line in enumerate(image_lines):
        if len(line) != width:
            raise ValueError(
                f"Image file line {i} has length {len(line)}, "
                f"expected {width}."
            )

    # ----------------------------
    # 4. Build Sample objects
    # ----------------------------
    samples: List[Sample] = []
    for i in range(num_images):
        start = i * height
        end = start + height
        img_rows = image_lines[start:end]
        samples.append(Sample(pixels=img_rows, label=labels[i]))

    return samples, height, width
