"""
src/predict.py
Runs a trained model on a single image file (PNG/JPG of a handwritten digit)
and prints the predicted digit with confidence.

Usage:
    python src/predict.py --model cnn --image my_digit.png
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

from data_loader import preprocess_single_image
from preprocess import normalize_digit

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def load_and_prepare_image(path: str) -> np.ndarray:
    """
    Loads any image of a handwritten digit and converts it to MNIST format:
    grayscale, white strokes on black (photos are usually dark-on-light, so
    it is inverted when needed), paper/background removed, then cropped,
    scaled to 20 px and centred in 28x28 exactly the way MNIST digits are
    (see preprocess.py). Returns a 28x28 float array in 0-1.
    """
    arr = np.array(Image.open(path).convert("L")).astype("float32")
    if arr.mean() > 127:           # dark ink on light paper -> flip to MNIST's convention
        arr = 255 - arr
    bg = np.percentile(arr, 50)    # the paper: most pixels are background
    v = np.clip((arr - bg) / (arr.max() - bg + 1e-6), 0, 1)
    v[v < 0.15] = 0                # drop paper texture / faint noise
    out = normalize_digit(v)
    if out is None:
        raise SystemExit("Couldn't find any digit strokes in that image.")
    return out


def main():
    parser = argparse.ArgumentParser(description="Predict the digit in an image using a trained model.")
    parser.add_argument("--model", choices=["ann", "cnn"], default="cnn")
    parser.add_argument("--image", required=True, help="Path to a PNG/JPG image of a handwritten digit.")
    args = parser.parse_args()

    model_path = os.path.join(SAVED_MODELS_DIR, f"{args.model}_model.keras")
    model = load_model(model_path)

    arr = load_and_prepare_image(args.image)
    inputs = preprocess_single_image(arr)
    x = inputs[args.model]

    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    print(f"Predicted digit: {pred}  (confidence: {confidence:.2%})")
    print("Full probability distribution:")
    for digit, p in enumerate(probs):
        bar = "#" * int(p * 40)
        print(f"  {digit}: {p:.4f} {bar}")


if __name__ == "__main__":
    main()
