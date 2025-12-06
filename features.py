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


def pixel_features(sample: Sample) -> List[int]:
    """
    Converts ASCII pixels to a binary vector.
      - '#' or '+' → 1
      - ' '         → 0
    """
    vec = []
    for row in sample.pixels:
        for ch in row:
            vec.append(0 if ch == " " else 1)
    return vec


def counting_features(sample: Sample) -> List[int]:
    """
    Counts:
       - filled pixels (# or +)
       - empty pixels (spaces)
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
