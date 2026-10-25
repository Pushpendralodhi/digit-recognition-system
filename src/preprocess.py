"""
src/preprocess.py
MNIST-style normalisation for anything a person draws.

Why this exists: MNIST digits are not "a digit somewhere on a 28x28 canvas".
Every MNIST image was cropped to the digit's bounding box, scaled so the
longer side is 20 pixels, and then centred in the 28x28 frame by its centre
of mass. A model trained on that never sees a digit that is small, huge or
off-centre — so if you squash a raw 280x280 drawing straight down to 28x28,
anything that isn't drawn big and dead-centre gets misread.

normalize_digit() reproduces the MNIST recipe, and the browser version in
webapp_static/index.html implements the same algorithm step for step.
"""

import numpy as np

# Mean centre of mass of the real MNIST training set (row, col), measured on
# the training set. Used as the target so drawn digits land where MNIST's do.
COM_TARGET = (14.0, 14.0)
BOX = 20          # longer side of the digit after scaling
OUT = 28          # final image size
BBOX_THRESHOLD = 0.1   # pixels above 10% of the max count as "ink"


def _resize_matrix(src_len: int, dst_len: int) -> np.ndarray:
    """(dst_len x src_len) weights: area-average when shrinking, linear when growing."""
    s = src_len / dst_len
    W = np.zeros((dst_len, src_len), dtype=np.float32)
    if s >= 1.0:  # shrinking: each output pixel = average of the source span it covers
        for i in range(dst_len):
            lo, hi = i * s, (i + 1) * s
            j0, j1 = int(np.floor(lo)), int(np.ceil(hi))
            for j in range(j0, min(j1, src_len)):
                W[i, j] = max(0.0, min(hi, j + 1) - max(lo, j))
            W[i] /= W[i].sum()
    else:         # growing: linear interpolation between source pixel centres
        for i in range(dst_len):
            c = (i + 0.5) * s - 0.5
            j0 = int(np.floor(c))
            f = c - j0
            a, b = min(max(j0, 0), src_len - 1), min(max(j0 + 1, 0), src_len - 1)
            W[i, a] += 1 - f
            W[i, b] += f
    return W


def resize_area(img: np.ndarray, dh: int, dw: int) -> np.ndarray:
    h, w = img.shape
    return _resize_matrix(h, dh) @ img @ _resize_matrix(w, dw).T


def normalize_digit(img: np.ndarray):
    """
    img: 2-D float array, any size, values 0-1, digit = bright strokes on dark.
    Returns a (28, 28) float32 array in MNIST's format, or None if blank.
    """
    img = np.asarray(img, dtype=np.float32)
    m = float(img.max())
    if m <= 0.05:
        return None
    ink = img > BBOX_THRESHOLD * m
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    crop = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    h, w = crop.shape
    scale = BOX / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = resize_area(crop, nh, nw)

    out = np.zeros((OUT, OUT), dtype=np.float32)
    top, left = (OUT - nh) // 2, (OUT - nw) // 2
    out[top:top + nh, left:left + nw] = small

    # translate so the centre of mass sits where MNIST's does
    total = out.sum()
    if total > 0:
        ys, xs = np.mgrid[0:OUT, 0:OUT]
        cy, cx = (out * ys).sum() / total, (out * xs).sum() / total
        dy, dx = int(round(COM_TARGET[0] - cy)), int(round(COM_TARGET[1] - cx))
        shifted = np.zeros_like(out)
        ys0, ys1 = max(0, dy), min(OUT, OUT + dy)
        xs0, xs1 = max(0, dx), min(OUT, OUT + dx)
        shifted[ys0:ys1, xs0:xs1] = out[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        out = shifted
    return np.clip(out, 0.0, 1.0)


def naive_resize(img: np.ndarray) -> np.ndarray:
    """The OLD behaviour (squash whole canvas to 28x28) — kept only for comparison."""
    return np.clip(resize_area(np.asarray(img, dtype=np.float32), OUT, OUT), 0.0, 1.0)


# --------------------------------------------------------------------------
# Pen normalisation (the step that makes mouse/touchpad drawings reliable)
# --------------------------------------------------------------------------
# A canvas drawing is a constant-width pen line, but its width *relative to the
# digit* changes with how big you draw: a small digit drawn with the same pen
# looks fat, its loops fill in and a 6 turns into a blob that reads as 0.
# MNIST digits always have the same slim stroke. So before recognising we
#   1. scale the drawing to a fixed working size,
#   2. thin every stroke down to a 1-pixel centre line (Zhang-Suen thinning),
#   3. redraw that centre line with one fixed MNIST-like pen,
# which makes pen size and digit size irrelevant. The browser app implements
# exactly the same steps (webapp_static/index.html: penNormalize()).

WORK = 96       # working resolution (longer side) for thinning
PEN = 7         # pen diameter in working pixels (~ MNIST stroke width after scaling)


def _zs_tables():
    """Look-up tables for the two Zhang-Suen sub-iterations, indexed by the 3x3 neighbourhood
    packed into 9 bits (bit i = neighbour P(i+2) in the order N, NE, E, SE, S, SW, W, NW; bit 8 = centre)."""
    tabs = np.zeros((2, 512), dtype=bool)
    for code in range(512):
        nb = [(code >> i) & 1 for i in range(8)]
        P2, P3, P4, P5, P6, P7, P8, P9 = nb
        centre = (code >> 8) & 1
        B = sum(nb)
        A = sum(1 for i in range(8) if nb[i] == 0 and nb[(i + 1) % 8] == 1)
        base = centre == 1 and 2 <= B <= 6 and A == 1
        tabs[0, code] = base and (P2 * P4 * P6 == 0) and (P4 * P6 * P8 == 0)
        tabs[1, code] = base and (P2 * P4 * P8 == 0) and (P2 * P6 * P8 == 0)
    return tabs


_ZS_TABS = _zs_tables()
_ZS_KERNEL = np.zeros((3, 3), dtype=np.int32)
for _i, (_dr, _dc) in enumerate([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]):
    _ZS_KERNEL[1 + _dr, 1 + _dc] = 1 << _i
_ZS_KERNEL[1, 1] = 1 << 8


def zhang_suen_thin(binary: np.ndarray) -> np.ndarray:
    """Classic Zhang-Suen thinning (table-driven, same result as the textbook loop).
    binary: 2-D bool/0-1 array -> thinned uint8 0/1."""
    from scipy import ndimage as ndi
    img = np.pad(np.asarray(binary, dtype=np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            codes = ndi.correlate(img.astype(np.int32), _ZS_KERNEL, mode="constant", cval=0)
            kill = _ZS_TABS[step][codes]
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1]


def disc_dilate(binary: np.ndarray, diameter: int) -> np.ndarray:
    """Redraw every 1-pixel of `binary` as a filled disc of the given diameter."""
    r = diameter / 2.0
    ri = int(np.ceil(r))
    h, w = binary.shape
    padded = np.pad(np.asarray(binary, dtype=np.uint8), ri)
    out = np.zeros((h, w), dtype=np.uint8)
    for dy in range(-ri, ri + 1):
        for dx in range(-ri, ri + 1):
            if dy * dy + dx * dx <= (r - 0.5) ** 2 + 0.25 or (dy == 0 and dx == 0):
                out |= padded[ri + dy:ri + dy + h, ri + dx:ri + dx + w]
    return out


def pen_normalize(img: np.ndarray):
    """
    Canvas drawing (2-D float 0-1, bright strokes on dark) -> slim, fixed-width
    version of the same drawing at WORK resolution (float32 0/1), or None if blank.
    """
    img = np.asarray(img, dtype=np.float32)
    m = float(img.max())
    if m <= 0.05:
        return None
    ink = img > BBOX_THRESHOLD * m
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    crop = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    s = WORK / max(h, w)
    nh, nw = max(3, int(round(h * s))), max(3, int(round(w * s)))
    big = resize_area(crop, nh, nw) > 0.5 * m
    thin = zhang_suen_thin(big)
    return disc_dilate(thin, PEN).astype(np.float32)


def normalize_pen_digit(img: np.ndarray):
    """Full pipeline for a drawn digit: pen_normalize -> MNIST crop/scale/centre -> (28,28) float32."""
    slim = pen_normalize(img)
    return None if slim is None else normalize_digit(slim)
