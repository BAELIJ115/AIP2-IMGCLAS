# naive_bayes.py
# implements naive bayes classifier 

import math
from typing import List


class NaiveBayesClassifier:

    def __init__(self):
        self.num_classes = None
        self.num_features = None

        # p(class)
        self.class_priors = []

        # p(feature_i = 1 | class = c)
        # Shape: [num_classes][num_features]
        self.feature_probs = []

    # training
    def fit(self, X: List[List[int]], y: List[int]):
        """
        Fit the Naive Bayes classifier.

        X : list of feature vectors (each a list of ints: 0/1 or small int)
        y : list of class labels (e.g., 0–9 for digits, 0/1 for faces)
        """
        n_samples = len(X)
        self.num_features = len(X[0])
        self.num_classes = len(set(y))

        # examples per class
        class_counts = [0] * self.num_classes

        # keep track of feature counts per class
        feature_counts = [
            [0] * self.num_features
            for _ in range(self.num_classes)
        ]
        # counting occurences
        for feats, label in zip(X, y):
            class_counts[label] += 1

            for i, value in enumerate(feats):
                if value > 0:  # treat positive number as "on"
                    feature_counts[label][i] += 1

        # compute priors: p(class = c)
        self.class_priors = [
            class_counts[c] / n_samples
            for c in range(self.num_classes)
        ]

        # conditional probabilities:
        # p(feature_i = 1 | class = c)
        self.feature_probs = [
            [0] * self.num_features
            for _ in range(self.num_classes)
        ]

        for c in range(self.num_classes):
            for i in range(self.num_features):
                count_on = feature_counts[c][i]
                total = class_counts[c]
                prob = (count_on + 1) / (total + 2)
                self.feature_probs[c][i] = prob

 
    # predict one sample
    def predict_one(self, feats: List[int]) -> int:
        best_class = None
        best_score = -math.inf

        for c in range(self.num_classes):
            score = math.log(self.class_priors[c])

            for i, value in enumerate(feats):
                p = self.feature_probs[c][i]

                if value > 0:       # feature on
                    score += math.log(p)
                else:               # feature off
                    score += math.log(1 - p)

            # tracks best scoring class
            if score > best_score:
                best_score = score
                best_class = c

        return best_class

    def predict(self, X: List[List[int]]) -> List[int]:
        return [self.predict_one(feats) for feats in X]
