"""
scripts/robustness_check.py
Measures how well a saved model handles digits that are NOT clean MNIST
test images — i.e. what actually happens when a person draws on the canvas.

Stress sets (every one is a 280x280 canvas drawn with the app's 18 px pen, or typeset):
  1. simulated mouse/touchpad drawings (src/canvas_sim.py) - same generator the
     training set used (different random draws), so this is the *easiest* set
  2. HELD-OUT stroke fonts (plain / complex / script Hershey fonts) - traced with the
     pen at random size, position and tilt; these fonts were never trained on
  3. typeset digits from matplotlib's bundled TTF fonts - never trained on

Each is scored with three pipelines: SQUASH (the original: shrink the whole canvas to
28x28), MNIST-CROP (crop/scale/centre) and PEN (what the app now does: thin the
strokes to a fixed-width pen line first, then crop/scale/centre).

Usage:
    python scripts/robustness_check.py --model saved_models/cnn_model.keras
"""
import argparse, glob, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from canvas_sim import make_set, CANVAS
from preprocess import normalize_digit, normalize_pen_digit, naive_resize
from font_sim import font_canvas, HELD_OUT_FONTS


def font_set(n_per_digit=60, seed=7):
    import matplotlib
    fonts = sorted(glob.glob(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "*.ttf")))
    fonts = [f for f in fonts if "Symbol" not in f and "Math" not in f and "Display" not in f and "cm" not in os.path.basename(f)[:2]]
    rng = np.random.default_rng(seed)
    X, y = [], []
    for d in range(10):
        for _ in range(n_per_digit):
            size = int(rng.uniform(90, 250))
            font = ImageFont.truetype(fonts[rng.integers(len(fonts))], size)
            img = Image.new("L", (CANVAS, CANVAS), 0)
            dr = ImageDraw.Draw(img)
            bb = dr.textbbox((0, 0), str(d), font=font)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            x = rng.uniform(5, max(6, CANVAS - w - 5)) - bb[0]
            yy = rng.uniform(5, max(6, CANVAS - h - 5)) - bb[1]
            dr.text((x, yy), str(d), fill=255, font=font)
            X.append(np.asarray(img, dtype=np.float32) / 255.0); y.append(d)
    return np.stack(X), np.array(y)


def stroke_font_set(n_per_digit=100, seed=99):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for d in range(10):
        for i in range(n_per_digit):
            X.append(font_canvas(d, HELD_OUT_FONTS[i % len(HELD_OUT_FONTS)], rng)); y.append(d)
    return np.stack(X), np.array(y)


def score(predict, X, y, prep):
    batch, keep = [], []
    for i, img in enumerate(X):
        p = prep(img)
        if p is not None:
            batch.append(p); keep.append(i)
    probs = predict(np.stack(batch))
    return float((np.argmax(probs, 1) == y[keep]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100, help="simulated drawings per digit")
    ap.add_argument("--json", help="write results to this file")
    a = ap.parse_args()

    import tensorflow as tf
    model = tf.keras.models.load_model(a.model)
    flat = len(model.input_shape) == 2
    predict = lambda B: model.predict(B.reshape(len(B), 784) if flat else B.reshape(len(B), 28, 28, 1), verbose=0)

    sets = {"simulated mouse drawings": make_set(a.n, seed=123),
            "held-out stroke fonts": stroke_font_set(),
            "typeset digits (TTF fonts)": font_set()}
    results = {}
    print(f"\nModel: {a.model}\n")
    print(f"{'test set':32s} {'squash':>9s} {'mnist-crop':>11s} {'pen (app)':>10s}")
    for name, (X, y) in sets.items():
        sq = score(predict, X, y, naive_resize)
        cr = score(predict, X, y, normalize_digit)
        pen = score(predict, X, y, normalize_pen_digit)
        results[name] = {"squash": sq, "mnist_crop": cr, "pen": pen, "n": int(len(y))}
        print(f"{name:32s} {sq*100:8.1f}% {cr*100:10.1f}% {pen*100:9.1f}%")
    if a.json:
        json.dump(results, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
