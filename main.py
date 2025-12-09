# main.py

import argparse
import os
import random

from data_loader import load_ascii_dataset
from features import extract_features
from naive_bayes import NaiveBayesClassifier
from perceptron import PerceptronClassifier
from evaluation import run_subsample_experiments


def build_default_paths(data_dir: str, dataset: str):
    """
    Construct the file paths for training/validation/test sets.
    """
    dataset = dataset.lower()

    if dataset == "digits":
        subdir = "digitdata"
        train_img = "trainingimages"
        train_lbl = "traininglabels"
        val_img = "validationimages"
        val_lbl = "validationlabels"
        test_img = "testimages"
        test_lbl = "testlabels"

    elif dataset == "faces":
        subdir = "facedata"
        train_img = "facedatatrain"
        train_lbl = "facedatatrainlabels"
        val_img = "facedatavalidation"
        val_lbl = "facedatavalidationlabels"
        test_img = "facedatatest"
        test_lbl = "facedatatestlabels"

    else:
        raise ValueError("Dataset must be 'digits' or 'faces'.")

    base = os.path.join(data_dir, subdir)

    return {
        "train_images": os.path.join(base, train_img),
        "train_labels": os.path.join(base, train_lbl),
        "val_images": os.path.join(base, val_img),
        "val_labels": os.path.join(base, val_lbl),
        "test_images": os.path.join(base, test_img),
        "test_labels": os.path.join(base, test_lbl),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["digits", "faces"],
        required=True
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["naive_bayes", "perceptron"],
        required=True
    )

    parser.add_argument(
        "--feature-type",
        type=str,
        default="pixels",
        choices=["pixels", "counting"]
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="."
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()
    random.seed(args.seed)
    # nicely structred header
    print("\n=== CS4346 Project 2 - Image Classifier ===\n")
    print(f"Dataset:     {args.dataset}")
    print(f"Algorithm:   {args.algorithm}")
    print(f"Features:    {args.feature_type}\n")

    # Load dataset
    paths = build_default_paths(args.data_dir, args.dataset)
    #building dictionary of file paths using data dir and dataset type

    train_samples, h_tr, w_tr = load_ascii_dataset(paths["train_images"], paths["train_labels"]) #returns list of Sample objects and image dimensions

    _, h_val, w_val = load_ascii_dataset(paths["val_images"], paths["val_labels"])

    test_samples, h_tst, w_tst = load_ascii_dataset(paths["test_images"], paths["test_labels"])

    # Ensure consistent image size
    assert (h_tr, w_tr) == (h_val, w_val) == (h_tst, w_tst), "Image sizes mismatch!"

    print(f"Train size:  {len(train_samples)}")
    print(f"Test size:   {len(test_samples)}\n")

    # Extract features
    X_train = extract_features(train_samples, args.feature_type)
    y_train = [s.label for s in train_samples]

    X_test = extract_features(test_samples, args.feature_type)
    y_test = [s.label for s in test_samples]

    # Select classifier
    if args.algorithm == "naive_bayes":
        def factory():
            return NaiveBayesClassifier()
    else:
        def factory():
            return PerceptronClassifier(num_epochs=5, learning_rate=1.0)

    # Training percentages
    train_percentages = [0.1 * i for i in range(1, 11)]

    print("Running experiments...\n")

    # Run experiments
    results = run_subsample_experiments(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classifier_factory=factory,
        train_percentages=train_percentages,
        num_runs=args.runs
    )

    # Print results
    print("Results (TEST data):")
    print("Train%   MeanAcc   StdAcc    TrainTime(s)")
    for p in sorted(results.keys()):
        mean_acc, std_acc, mean_time = results[p]
        print(f"{int(p*100):>3d}%    {mean_acc:7.4f}   {std_acc:7.4f}   {mean_time:10.4f}")


if __name__ == "__main__":
    main()
