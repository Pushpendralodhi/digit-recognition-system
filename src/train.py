"""
src/train.py
Trains the ANN or the CNN, with early stopping, checkpointing and curves.

Usage:
    python src/train.py --model cnn --epochs 14
    python src/train.py --model ann --epochs 20
    python src/train.py --model cnn --plain-mnist        # old behaviour: raw MNIST only

By default the networks are trained on the "pen" training set built by
src/build_pen_training_set.py: real MNIST digits (raw, and re-drawn with the
same slim pen the web app applies to a drawing), simulated mouse-drawn digits
and stroke-font digits. That is what makes them reliable on digits people
actually draw with a mouse or touchpad, not just on MNIST test images.
The set is built automatically on the first run (a few minutes, ~100-250 MB).
"""

import argparse
import os
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from data_loader import load_mnist
from models import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(HERE, "..", "saved_models")
DEFAULT_DATA = os.path.join(HERE, "..", "work", "pen_train.npz")


def plot_history(history, model_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for key, title, fname in (("accuracy", "Accuracy", "accuracy"), ("loss", "Loss", "loss")):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[key], label=f"Training {title}", linewidth=2)
        plt.plot(history.history["val_" + key], label=f"Validation {title}", linewidth=2)
        plt.xlabel("Epoch"); plt.ylabel(title)
        plt.title(f"{model_name.upper()} — Training vs Validation {title}")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, f"{model_name}_{fname}.png"), dpi=150, bbox_inches="tight")
        plt.close()


def load_pen_set(path):
    if not os.path.exists(path):
        print(f"{path} not found - building the training set first (a few minutes)...")
        subprocess.check_call([sys.executable, os.path.join(HERE, "build_pen_training_set.py"),
                               "--out", os.path.dirname(path)])
    z = np.load(path)
    f = lambda a: a.astype("float32").reshape(-1, 28, 28) / 255.0
    return f(z["X_train"]), z["y_train"].astype(int), f(z["X_val"]), z["y_val"].astype(int)


def main():
    parser = argparse.ArgumentParser(description="Train the ANN or CNN digit recognizer.")
    parser.add_argument("--model", choices=["ann", "cnn"], required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data", default=DEFAULT_DATA, help="pen training set (.npz)")
    parser.add_argument("--plain-mnist", action="store_true", help="train on raw MNIST only")
    parser.add_argument("--out-dir", default=SAVED_MODELS_DIR)
    args = parser.parse_args()

    if args.plain_mnist:
        print("Loading MNIST (plain)...")
        d = load_mnist()
        X_train, y_train = d["X_train_cnn"].reshape(-1, 28, 28), d["y_train_int"]
        X_val, y_val = d["X_val_cnn"].reshape(-1, 28, 28), d["y_val_int"]
    else:
        X_train, y_train, X_val, y_val = load_pen_set(args.data)
    print(f"Training images: {len(X_train):,}   validation images: {len(X_val):,}")

    shape = (-1, 784) if args.model == "ann" else (-1, 28, 28, 1)
    X_train, X_val = X_train.reshape(shape), X_val.reshape(shape)
    y_train, y_val = to_categorical(y_train, 10), to_categorical(y_val, 10)

    model = build_model(args.model)
    model.summary()

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.out_dir, f"{args.model}_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=1, min_lr=1e-5),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]
    print(f"Training for up to {args.epochs} epochs (early stopping enabled)...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=args.epochs, batch_size=args.batch_size,
                        shuffle=True, callbacks=callbacks, verbose=2)

    plot_history(history, args.model, out_dir=os.path.join(args.out_dir, "..", "results"))
    print(f"\nBest model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
