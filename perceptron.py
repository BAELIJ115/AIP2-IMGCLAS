#the goal of this is to implement perceptron algorithm for classifying images of digits and faces
# perceptron.py
# perceptron.py

from typing import List
import random


class PerceptronClassifier:
    """
    Multi-class perceptron.
    Works for:
      - digits (10 classes)
      - faces (2 classes)
    """

    def __init__(self, num_epochs: int = 5, learning_rate: float = 1.0):
        self.num_epochs = num_epochs
        self.lr = learning_rate

        self.num_classes = None
        self.num_features = None

        # weights[class][feature_index]
        self.weights = None

    # ----------------------------------------------------
    # TRAINING
    # ----------------------------------------------------
    def fit(self, X: List[List[int]], y: List[int]):
        """
        X = list of feature vectors
        y = list of labels
        """

        n_samples = len(X)
        self.num_features = len(X[0])
        self.num_classes = len(set(y))

        # Initialize all weights to zero
        # shape: [num_classes][num_features]
        self.weights = [
            [0.0] * self.num_features
            for _ in range(self.num_classes)
        ]

        # Training loop
        for epoch in range(self.num_epochs):
            # shuffle the order each epoch
            indices = list(range(n_samples))
            random.shuffle(indices)

            for idx in indices:
                feats = X[idx]
                true_label = y[idx]

                # Predict using current weights
                predicted = self.predict_one(feats)

                # Update rule only if wrong
                if predicted != true_label:
                    for i in range(self.num_features):
                        # reward true class
                        self.weights[true_label][i] += self.lr * feats[i]
                        # punish wrong class
                        self.weights[predicted][i] -= self.lr * feats[i]

    # ----------------------------------------------------
    # PREDICT a single example
    # ----------------------------------------------------
    def predict_one(self, feats: List[int]) -> int:
        """
        Compute dot product with each class's weight vector.
        Return the class with highest score.
        """
        best_class = None
        best_score = float("-inf")

        for c in range(self.num_classes):
            score = 0.0
            w = self.weights[c]

            # dot-product manually
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
