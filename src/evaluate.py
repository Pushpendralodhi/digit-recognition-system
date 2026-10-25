"""
src/evaluate.py
Evaluates a trained model on the MNIST test set: accuracy, precision/recall/F1
per digit, and a confusion matrix. Can also compare ANN vs CNN side by side.

Usage:
    python src/evaluate.py --model ann
    python src/evaluate.py --model cnn
    python src/evaluate.py --compare        # evaluates both, side-by-side bar chart
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

from data_loader import load_mnist

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def evaluate_one(kind: str, data: dict):
    model_path = os.path.join(SAVED_MODELS_DIR, f"{kind}_model.keras")
    model = load_model(model_path)

    X_test = data["X_test_ann"] if kind == "ann" else data["X_test_cnn"]
    y_test_int = data["y_test_int"]

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    acc = (preds == y_test_int).mean()
    print(f"\n===== {kind.upper()} =====")
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test_int, preds, digits=4))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_test_int, preds)
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{kind.upper()} — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{kind}_confusion_matrix.png"), dpi=150)
    plt.close()

    return acc, preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on the MNIST test set.")
    parser.add_argument("--model", choices=["ann", "cnn"])
    parser.add_argument("--compare", action="store_true", help="Evaluate both ANN and CNN and compare.")
    args = parser.parse_args()

    if not args.model and not args.compare:
        parser.error("Pass --model ann|cnn, or --compare to evaluate both.")

    data = load_mnist()

    if args.compare:
        ann_acc, _ = evaluate_one("ann", data)
        cnn_acc, _ = evaluate_one("cnn", data)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.figure(figsize=(6, 5))
        bars = plt.bar(["ANN", "CNN"], [ann_acc, cnn_acc], color=["#2c6e9e", "#a83232"])
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1)
        plt.title("ANN vs CNN — Test Accuracy Comparison")
        for b, acc in zip(bars, [ann_acc, cnn_acc]):
            plt.text(b.get_x() + b.get_width() / 2, acc + 0.02, f"{acc:.4f}", ha="center", fontweight="bold")
        plt.savefig(os.path.join(RESULTS_DIR, "ann_vs_cnn_comparison.png"), dpi=150, bbox_inches="tight")
        print(f"\nComparison chart saved to {RESULTS_DIR}/ann_vs_cnn_comparison.png")
    else:
        evaluate_one(args.model, data)


if __name__ == "__main__":
    main()
