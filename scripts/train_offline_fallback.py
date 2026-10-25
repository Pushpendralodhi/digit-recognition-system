"""
scripts/train_offline_fallback.py
Trains a small scikit-learn MLP on the offline 8x8 digits dataset (no
internet or TensorFlow required) and saves it so the Flask app has
*something* real to serve predictions from immediately.

This is a convenience fallback only. For the real project deliverable,
run src/train.py on a machine/Colab with TensorFlow and internet access
to train on the actual 28x28 MNIST dataset — the app will automatically
prefer that model if it's present in saved_models/.
"""

import os
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def main():
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        alpha=1e-4, max_iter=200, random_state=42,
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Offline fallback model trained. Test accuracy: {acc:.4f}")

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    out_path = os.path.join(SAVED_MODELS_DIR, "ann_sklearn_demo.joblib")
    joblib.dump(clf, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
