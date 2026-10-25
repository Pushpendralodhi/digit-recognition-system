"""
src/font_sim.py
Extra training/test digits drawn from OpenCV's Hershey *stroke* fonts, traced with the
canvas's 18 px round pen at random size, position and tilt. Typeset digits have shapes
MNIST's writers rarely use (a 1 with a flag and base, a 7 with a bar, a closed 4...) which
real people do write, so they widen what the model has seen.

TRAIN_FONTS are used for training. HELD_OUT_FONTS are never trained on: they are the
honest robustness test (scripts/robustness_check.py).
"""
import cv2
import numpy as np

_F = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX, "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX, "simplex_italic": cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,
    "plain": cv2.FONT_HERSHEY_PLAIN, "complex": cv2.FONT_HERSHEY_COMPLEX,
    "script": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}
TRAIN_FONTS = ["simplex", "duplex", "triplex", "simplex_italic"]
HELD_OUT_FONTS = ["plain", "complex", "script"]


def font_canvas(digit, font, rng, canvas=280, S=3):
    h_t = rng.uniform(0.5, 0.95) * canvas * S
    (tw, th), base = cv2.getTextSize(str(digit), _F[font], 8.0, 1)
    tmp = np.zeros((th + base + 60, tw + 60), np.uint8)
    cv2.putText(tmp, str(digit), (30, th + 30), _F[font], 8.0, 255, 1, cv2.LINE_AA)
    ys, xs = np.where(tmp > 0)
    tmp = tmp[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = h_t / tmp.shape[0]
    w_t = tmp.shape[1] * f * rng.uniform(0.8, 1.15)
    tmp = cv2.resize(tmp, (max(2, int(w_t)), max(2, int(h_t))), interpolation=cv2.INTER_AREA)
    tmp = (tmp > 40).astype(np.uint8) * 255
    lw = int(rng.uniform(14, 22) * S)                       # the app's pen is 18 px
    tmp = cv2.dilate(tmp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lw, lw)))
    M = cv2.getRotationMatrix2D((tmp.shape[1] / 2, tmp.shape[0] / 2), rng.normal(0, 7), 1.0)
    tmp = cv2.warpAffine(tmp, M, (tmp.shape[1] + 20, tmp.shape[0] + 20))
    th2, tw2 = tmp.shape
    if th2 >= canvas * S or tw2 >= canvas * S:
        tmp = cv2.resize(tmp, (min(tw2, canvas * S - 10), min(th2, canvas * S - 10)))
        th2, tw2 = tmp.shape
    big = np.zeros((canvas * S, canvas * S), np.uint8)
    y0, x0 = rng.integers(0, canvas * S - th2 + 1), rng.integers(0, canvas * S - tw2 + 1)
    big[y0:y0 + th2, x0:x0 + tw2] = tmp
    return cv2.resize(big, (canvas, canvas), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def _worker(args):
    from preprocess import normalize_pen_digit
    digit, fonts, n, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a = normalize_pen_digit(font_canvas(digit, fonts[i % len(fonts)], rng))
        out.append(a if a is not None else np.zeros((28, 28), np.float32))
    return np.stack(out), np.full(n, digit)


def make_font_set(n_per_digit, fonts=None, seed=0, workers=2):
    """Pen-normalised font digits -> (N,28,28) float32, labels (N,)."""
    from concurrent.futures import ProcessPoolExecutor
    fonts = fonts or TRAIN_FONTS
    jobs = [(d, fonts, n_per_digit // 4, seed * 6151 + d * 17 + k) for d in range(10) for k in range(4)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_worker, jobs))
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])
