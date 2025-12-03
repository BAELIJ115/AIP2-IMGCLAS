# the goal of this file is to subsample training data, run several random trials, train the soecific classifier,meaure accuracy and runtme, and give avgerages of these metrics
# evaluation.py

# evaluation.py

import random
import time
from statistics import mean, stdev
from typing import List, Callable, Dict, Tuple


def accuracy(preds: List[int], labels: List[int]) -> float:
    """
    Compute simple classification accuracy.
    """
    correct = sum(p == y for p, y in zip(preds, labels))
    return correct / len(labels)


def run_subsample_experiments(
    X_train: List[List[int]],
    y_train: List[int],
    X_test: List[List[int]],
    y_test: List[int],
    classifier_factory: Callable[[], object],
    train_percentages: List[float],
    num_runs: int = 5
) -> Dict[float, Tuple[float, float, float]]:
    """
    For each train percentage:
        - randomly sample training data
        - train classifier
        - test accuracy
        - measure training time
        - repeat num_runs times

    Returns:
        results[p] = (mean_accuracy, std_accuracy, mean_train_time)
    """
    results = {}

    n_train_total = len(X_train)

    for p in train_percentages:
        print(f"Running experiments for {int(p * 100)}% training data...")

        run_accuracies = []
        run_times = []

        subset_size = max(1, int(p * n_train_total))

        for _ in range(num_runs):

            # ---------------------------------------------
            # Random subset of training data
            # ---------------------------------------------
            indices = list(range(n_train_total))
            random.shuffle(indices)
            chosen = indices[:subset_size]

            X_sub = [X_train[i] for i in chosen]
            y_sub = [y_train[i] for i in chosen]

            # ---------------------------------------------
            # Train the classifier
            # ---------------------------------------------
            clf = classifier_factory()

            t0 = time.time()
            clf.fit(X_sub, y_sub)
            train_time = time.time() - t0

            # ---------------------------------------------
            # Evaluate on the test set
            # ---------------------------------------------
            preds = clf.predict(X_test)
            acc = accuracy(preds, y_test)

            run_accuracies.append(acc)
            run_times.append(train_time)

        # -------------------------------------------------
        # Compute statistics
        # -------------------------------------------------
        mean_acc = mean(run_accuracies)
        std_acc = stdev(run_accuracies) if num_runs > 1 else 0.0
        mean_time = mean(run_times)

        results[p] = (mean_acc, std_acc, mean_time)

    return results
