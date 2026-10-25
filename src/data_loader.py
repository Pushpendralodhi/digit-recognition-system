"""
src/data_loader.py
Loads and preprocesses the real MNIST dataset for both the ANN and CNN.

Requires: tensorflow, scikit-learn, numpy

Data source: by default this loads the real MNIST idx-ubyte files from
data/raw/ (the original 60,000 train / 10,000 test images, downloaded once
and verified against the official MD5 checksums — see data/raw/SOURCE.md).
This avoids needing internet access every run and works the same on Colab
or any other machine. If data/raw/ is missing, it automatically falls back
to tf.keras.datasets.mnist.load_data(), which downloads MNIST itself
(needs internet).
"""

import gzip
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    # idx3 header: magic(4) num_images(4) rows(4) cols(4), big-endian uint32
    num = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16)
    return arr.reshape(num, rows, cols)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    # idx1 header: magic(4) num_labels(4), big-endian uint32
    arr = np.frombuffer(data, dtype=np.uint8, offset=8)
    return arr


def _load_from_local_files():
    files = {
        "train_images": os.path.join(_RAW_DIR, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(_RAW_DIR, "train-labels-idx1-ubyte.gz"),
        "test_images": os.path.join(_RAW_DIR, "t10k-images-idx3-ubyte.gz"),
        "test_labels": os.path.join(_RAW_DIR, "t10k-labels-idx1-ubyte.gz"),
    }
    if not all(os.path.exists(p) for p in files.values()):
        return None
    X_train = _read_idx_images(files["train_images"])
    y_train = _read_idx_labels(files["train_labels"])
    X_test = _read_idx_images(files["test_images"])
    y_test = _read_idx_labels(files["test_labels"])
    return (X_train, y_train), (X_test, y_test)


def load_mnist(val_split: float = 0.1, seed: int = 42):
    """
    Loads MNIST and returns everything both models need:

    Returns
    -------
    dict with keys:
        X_train_ann, X_val_ann, X_test_ann   -> flat (N, 784) float32 in [0,1]
        X_train_cnn, X_val_cnn, X_test_cnn   -> (N, 28, 28, 1) float32 in [0,1]
        y_train, y_val, y_test               -> one-hot (N, 10)
        y_train_int, y_val_int, y_test_int   -> integer labels (N,), for metrics/plots
    """
    local = _load_from_local_files()
    if local is not None:
        (X_train, y_train), (X_test, y_test) = local
    else:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train_ann_full = X_train.reshape(-1, 28 * 28)
    X_test_ann = X_test.reshape(-1, 28 * 28)
    X_train_cnn_full = X_train.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test.reshape(-1, 28, 28, 1)

    y_train_cat_full = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Same split indices reused for both ANN/CNN views so ANN and CNN are
    # trained/validated on exactly the same images.
    idx_train, idx_val = train_test_split(
        np.arange(len(X_train)), test_size=val_split, random_state=seed, stratify=y_train
    )

    return {
        "X_train_ann": X_train_ann_full[idx_train],
        "X_val_ann": X_train_ann_full[idx_val],
        "X_test_ann": X_test_ann,
        "X_train_cnn": X_train_cnn_full[idx_train],
        "X_val_cnn": X_train_cnn_full[idx_val],
        "X_test_cnn": X_test_cnn,
        "y_train": y_train_cat_full[idx_train],
        "y_val": y_train_cat_full[idx_val],
        "y_test": y_test_cat,
        "y_train_int": y_train[idx_train],
        "y_val_int": y_train[idx_val],
        "y_test_int": y_test,
    }


def preprocess_single_image(image_array: np.ndarray) -> dict:
    """
    Preprocesses one user-supplied image (already 28x28 grayscale, 0-255
    or 0-1) into the two shapes the models expect. Used by predict.py and
    the Flask app.
    """
    img = image_array.astype("float32")
    if img.max() > 1.0:
        img = img / 255.0
    img = img.reshape(28, 28)
    return {
        "ann": img.reshape(1, 28 * 28),
        "cnn": img.reshape(1, 28, 28, 1),
    }


if __name__ == "__main__":
    data = load_mnist()
    print("Train (ANN):", data["X_train_ann"].shape)
    print("Val   (ANN):", data["X_val_ann"].shape)
    print("Test  (ANN):", data["X_test_ann"].shape)
    print("Train (CNN):", data["X_train_cnn"].shape)
