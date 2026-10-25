"""
src/augment.py
Training-time data augmentation that mimics how real people draw digits
differently from MNIST's writers: tilted, slanted, squashed or stretched,
shaky lines, and thicker or thinner pen strokes.

Every augmented image is re-normalised with preprocess.normalize_digit(), so
the model is trained on exactly the kind of input the app feeds it.
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi

from preprocess import normalize_digit


def _augment_one(img, rng):
    # 1) random affine: rotation, shear, independent x/y scaling
    th = np.clip(rng.normal(0, 0.14), -0.35, 0.35)            # ~ +/-8 deg typical, 20 deg max
    sh = rng.uniform(-0.3, 0.3)
    sx, sy = rng.uniform(0.75, 1.3), rng.uniform(0.75, 1.3)
    c, s = np.cos(th), np.sin(th)
    A = np.array([[c, -s], [s, c]]) @ np.array([[1, sh], [0, 1]]) @ np.diag([sy, sx])
    M = np.linalg.inv(A)
    ctr = np.array([13.5, 13.5])
    img = ndi.affine_transform(img, M, offset=ctr - M @ ctr, order=1)

    # 2) shaky hand: smooth random displacement field
    if rng.random() < 0.5:
        alpha, sigma = rng.uniform(8, 24), rng.uniform(3, 4.5)
        dy = ndi.gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha
        dx = ndi.gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha
        yy, xx = np.mgrid[0:28, 0:28]
        img = ndi.map_coordinates(img, [yy + dy, xx + dx], order=1, mode="constant")

    # 3) pen thickness
    r = rng.random()
    if r < 0.30:
        img = ndi.grey_dilation(img, size=(2, 2))
    elif r < 0.45:
        img = ndi.grey_dilation(img, footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    elif r < 0.52:
        img = ndi.grey_dilation(img, size=(3, 3))
    elif r < 0.60:
        eroded = ndi.grey_erosion(img, size=(2, 2))
        if eroded.max() > 0.3:
            img = eroded / eroded.max()

    # 4) occasional soft edges
    if rng.random() < 0.25:
        img = ndi.gaussian_filter(img, rng.uniform(0.4, 0.8))

    out = normalize_digit(np.clip(img, 0, 1))
    if out is None:
        return None
    # 5) tiny leftover offset, so the model doesn't depend on pixel-perfect centring
    if rng.random() < 0.5:
        out = ndi.shift(out, rng.integers(-1, 2, 2), order=0, mode="constant")
    return out


def _worker(args):
    X, seed = args
    rng = np.random.default_rng(seed)
    out = np.empty_like(X)
    for i, img in enumerate(X):
        a = _augment_one(img, rng)
        out[i] = a if a is not None else img
    return out


def augment_dataset(X, copies=4, seed=0, workers=2):
    """X: (N, 28, 28) float32 in [0,1]. Returns (N*copies, 28, 28): copy k of image i
    is at index k*N + i, so labels are np.tile(y, copies)."""
    chunks, jobs = np.array_split(np.arange(len(X)), workers * 8), []
    for k in range(copies):
        for j, idx in enumerate(chunks):
            jobs.append((X[idx], seed * 100003 + k * 1009 + j))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_worker, jobs))
    out = []
    for k in range(copies):
        out.append(np.concatenate(res[k * len(chunks):(k + 1) * len(chunks)]))
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Pen-domain training data: MNIST digits pushed through the SAME pen
# normalisation the app applies to a drawing (preprocess.normalize_pen_digit),
# so the network is trained on exactly what it will be shown.
# --------------------------------------------------------------------------
import cv2
from preprocess import normalize_pen_digit


def distort_raw(img, rng):
    """Random tilt / slant / stretch + shaky-hand wobble on a 28x28 MNIST digit (no re-centring)."""
    th = np.clip(rng.normal(0, 0.14), -0.35, 0.35)
    sh = rng.uniform(-0.3, 0.3)
    sx, sy = rng.uniform(0.75, 1.3), rng.uniform(0.75, 1.3)
    c, s = np.cos(th), np.sin(th)
    A = np.array([[c, -s], [s, c]]) @ np.array([[1, sh], [0, 1]]) @ np.diag([sy, sx])
    M = np.linalg.inv(A)
    ctr = np.array([13.5, 13.5])
    img = ndi.affine_transform(img, M, offset=ctr - M @ ctr, order=1)
    if rng.random() < 0.5:
        alpha, sigma = rng.uniform(8, 24), rng.uniform(3, 4.5)
        dy = ndi.gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha
        dx = ndi.gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha
        yy, xx = np.mgrid[0:28, 0:28]
        img = ndi.map_coordinates(img, [yy + dy, xx + dx], order=1, mode="constant")
    return np.clip(img, 0, 1)


def _pen_worker(args):
    X, seed, distort = args
    rng = np.random.default_rng(seed)
    out = np.zeros((len(X), 28, 28), np.float32)
    for i, img in enumerate(X):
        im = distort_raw(img, rng) if distort else img
        up = np.clip(cv2.resize(im, (112, 112), interpolation=cv2.INTER_CUBIC), 0, 1)
        a = normalize_pen_digit(up)
        out[i] = a if a is not None else img
    return out


def pen_dataset(X, copies=1, seed=0, workers=2, distort=True):
    """X: (N,28,28) in [0,1] -> (N*copies, 28,28) pen-normalised, distorted copies (copy k of image i at k*N+i)."""
    chunks = np.array_split(np.arange(len(X)), workers * 8)
    jobs = [(X[idx], seed * 100003 + k * 1009 + j, distort) for k in range(copies) for j, idx in enumerate(chunks)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_pen_worker, jobs))
    return np.concatenate(res)
