# data_loader.py

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Sample:
    """
    Represents a single ASCII image + its label.
    - pixels: list of strings, each string is a row (with spaces and symbols)
    - label: int (digit 0–9 or face/not-face label)
    """
    pixels: List[str]
    label: int


def load_ascii_dataset(image_path: str, label_path: str) -> Tuple[List[Sample], int, int]:
    """
    Load the ASCII image dataset from text files.

    Returns:
        samples: list[Sample]
        height: int  (number of rows per image)
        width : int  (number of columns per image)
    """
    # --- 1. Load labels ---
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

    # --- 2. Load all image lines ---
    image_lines: List[str] = []
    with open(image_path, "r") as f:
        for line in f:
            # Keep spaces, only remove the newline at the end
            image_lines.append(line.rstrip("\n"))

    # Remove any trailing empty lines at the end of the file (just in case)
    while len(image_lines) > 0 and image_lines[-1] == "":
        image_lines.pop()

    total_lines = len(image_lines)
    if total_lines % num_images != 0:
        raise ValueError(
            f"Image file has {total_lines} lines but label file has "
            f"{num_images} labels. Cannot evenly divide into images."
        )

    # --- 3. Infer height and width ---
    height = total_lines // num_images
    width = len(image_lines[0]) if height > 0 else 0

    # Sanity check: ensure all lines have same width
    for i, line in enumerate(image_lines):
        if len(line) != width:
            raise ValueError(
                f"Line {i} in image file has length {len(line)}, "
                f"but expected {width}."
            )

    # --- 4. Build Sample objects ---
    samples: List[Sample] = []
    for i in range(num_images):
        start = i * height
        end = (i + 1) * height
        img_rows = image_lines[start:end]
        label = labels[i]
        samples.append(Sample(pixels=img_rows, label=label))

    return samples, height, width
