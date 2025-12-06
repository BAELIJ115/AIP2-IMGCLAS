# data_loader.py

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Sample:
    """
    Represents a single ASCII image + its label.
    - pixels: list of strings, each string is a row; label: int (digit 0–9 or face or not-face label)
    """
    pixels: List[str]
    label: int


def load_ascii_dataset(image_path: str, label_path: str) -> Tuple[List[Sample], int, int]:
    """
    Load the ASCII img dataset from text files and returns:
        samples: list[Sample]
        height: int  (number of rows)
        width : int  (number of columns)
    """
    # loading labels
    labels: List[int] = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            labels.append(int(line))

    num_images = len(labels)
    if num_images == 0:
        raise ValueError("No labels found in label file.")

    # loading image lines
    image_lines: List[str] = []
    with open(image_path, "r") as f:
        for line in f:
            # Keep spaces, only remove the newline at the end
            image_lines.append(line.rstrip("\n"))

    # removing empty lines (incase)
    while len(image_lines) > 0 and image_lines[-1] == "":
        image_lines.pop()

    total_lines = len(image_lines)
    if total_lines % num_images != 0:
        raise ValueError(
            f"Img file has {total_lines} lines but label file has "
            f"{num_images} labels. cant evenly divide."
        )

    # infer height and width
    height = total_lines // num_images
    width = len(image_lines[0]) if height > 0 else 0

    # ensure all lines have same width
    for i, line in enumerate(image_lines):
        if len(line) != width:
            raise ValueError(
                f"Line {i} in image file has length {len(line)}, "
                f"but expected {width}."
            )

    # building sample objects
    samples: List[Sample] = []
    for i in range(num_images):
        start = i * height
        end = (i + 1) * height
        img_rows = image_lines[start:end]
        label = labels[i]
        samples.append(Sample(pixels=img_rows, label=label))

    return samples, height, width
