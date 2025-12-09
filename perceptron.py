# perceptron.py
# Multi-class Perceptron Classifier for digits and faces

from typing import List
import random

class PerceptronClassifier:
    """
    Multi-class perceptron classifier.
    Supports:
      - digit classification (10 classes)
      - face classification (2 classes)
    """

    def __init__(self, num_epochs: int = 5, learning_rate: float = 1.0):
        self.num_epochs = num_epochs
        self.lr = learning_rate

        self.num_features = None
        self.num_classes = None
        self.weights = None  # shape: [num_classes][num_features]

    # ----------------------------------------------------
    # TRAINING
    # ----------------------------------------------------
    def fit(self, X: List[List[int]], y: List[int]):
        n_samples = len(X)
        self.num_features = len(X[0])
        self.num_classes = len(set(y))

        # Initialize all weights to zero
        self.weights = [
            [0.0] * self.num_features
            for _ in range(self.num_classes)
        ]

        # Training loop
        for epoch in range(self.num_epochs):
            indices = list(range(n_samples))
            random.shuffle(indices)

            for idx in indices:
                feats = X[idx]
                true_label = y[idx]

                # Predict with current weights
                predicted = self.predict_one(feats)

                # Update rule: reward true class, punish wrong class
                if predicted != true_label:
                    for i in range(self.num_features):
                        self.weights[true_label][i] += self.lr * feats[i]
                        self.weights[predicted][i] -= self.lr * feats[i]

    # ----------------------------------------------------
    # PREDICT a single example
    # ----------------------------------------------------
    def predict_one(self, feats: List[int]) -> int:
        best_class = None
        best_score = -float("inf")

        for c in range(self.num_classes):
            score = 0.0
            w = self.weights[c]

            # Compute dot product
            for i in range(self.num_features):
                score += w[i] * feats[i]

            if score > best_score:
                best_score = score
                best_class = c

        return best_class

    # ----------------------------------------------------
    # PREDICT many
    # ----------------------------------------------------
    def predict(self, X: List[List[int]]) -> List[int]:
        return [self.predict_one(feats) for feats in X]
