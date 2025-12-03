# the goal of this file is to  make functions that will take a sample from data loader and convert it into a list of numbers
# representing the features of the sample. These features will be used for training a machine learning model.
# we will implemnt 3 feature types (pixels,counting,and then both combined)

# features.py

from typing import List
from data_loader import Sample


def extract_features(samples: List[Sample], feature_type: str) -> List[List[int]]:
    """
    Apply the chosen feature extraction method to an entire dataset.
    Returns a list of feature vectors.
    """
    X = []

    for s in samples:
        if feature_type == "pixels":
            feats = pixel_features(s)

        elif feature_type == "counting":
            feats = counting_features(s)

        elif feature_type == "pixels+counting":
            feats = pixel_features(s) + counting_features(s)

        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        X.append(feats)

    return X


# -------------------------------------------------
# 1) Raw Pixel Features (binary)
# -------------------------------------------------
def pixel_features(sample: Sample) -> List[int]:
    """
    Converts ASCII pixels to binary vector.
    '#' or '+' → 1
    ' '        → 0
    """
    vec = []
    for row in sample.pixels:
        for ch in row:
            if ch == " ":
                vec.append(0)
            else:
                vec.append(1)
    return vec


# -------------------------------------------------
# 2) Counting Features (VERY simple)
# -------------------------------------------------
def counting_features(sample: Sample) -> List[int]:
    """
    Count:
       - number of filled pixels (# or +)
       - number of empty pixels (spaces)
    Returns a 2-dim vector [num_filled, num_empty]
    """
    filled = 0
    empty = 0

    for row in sample.pixels:
        for ch in row:
            if ch == " ":
                empty += 1
            else:
                filled += 1

    return [filled, empty]
