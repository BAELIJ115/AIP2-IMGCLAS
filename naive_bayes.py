#the goal of this file is to implement a Bernoulli Naive Bayes classifier for classifying images of digits and faces
# naive_bayes.py

import math
from typing import List


class NaiveBayesClassifier:
    """
    Bernoulli Naive Bayes for binary feature vectors.
    Works for both digits (10 classes) and faces (2 classes).
    """

    def __init__(self):
        self.num_classes = None
        self.num_features = None

        # Holds P(class)
        self.class_priors = []

        # Holds P(feature_i = 1 | class = c)
        self.feature_probs = []  # shape: [num_classes][num_features]

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def fit(self, X: List[List[int]], y: List[int]):
        """
        X: list of feature vectors (each is list of 0/1 values or small ints)
        y: list of labels
        """

        n_samples = len(X)
        self.num_features = len(X[0])
        self.num_classes = len(set(y))

        # Count how many examples per class
        class_counts = [0] * self.num_classes

        # Count feature occurrences:
        # feature_counts[c][i] = how many times feature i == 1 among examples of class c
        feature_counts = [
            [0] * self.num_features
            for _ in range(self.num_classes)
        ]

        # ------------------------------------------------
        # Pass 1: Count class occurrences & feature sums
        # ------------------------------------------------
        for feats, label in zip(X, y):
            class_counts[label] += 1
            for i, value in enumerate(feats):
                # If feature is binary or small integer: treat >0 as "on"
                if value > 0:
                    feature_counts[label][i] += 1

        # ------------------------------------------------
        # Compute priors: P(class = c)
        # ------------------------------------------------
        self.class_priors = [
            count / n_samples
            for count in class_counts
        ]

        # ------------------------------------------------
        # Compute conditional probabilities:
        # P(feature_i = 1 | class = c)
        # Using Laplace smoothing:
        # (count + 1) / (class_count + 2)
        # ------------------------------------------------
        self.feature_probs = [
            [0] * self.num_features
            for _ in range(self.num_classes)
        ]

        for c in range(self.num_classes):
            for i in range(self.num_features):
                count_on = feature_counts[c][i]
                total = class_counts[c]
                prob = (count_on + 1) / (total + 2)  # Laplace smoothing
                self.feature_probs[c][i] = prob

    # -------------------------------

