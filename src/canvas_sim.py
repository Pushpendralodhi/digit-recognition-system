"""
src/canvas_sim.py
Simulates what the web app's drawing canvas receives when a person draws a
digit with a mouse or touchpad: a 280x280 black canvas, a thick (18 px) white
round-capped stroke, wobbly hand-drawn paths, and digits of any size at any
position. Used ONLY to test robustness (scripts/robustness_check.py); the
training augmentation in augment.py is deliberately independent of this, so
the test is not graded on its own homework.

These are synthetic drawings, not real people — a useful stress test, not a
substitute for trying the app yourself.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

CANVAS = 280
LINE_WIDTH = 18


def _ellipse(cx, cy, rx, ry, a0=-90, a1=270, n=28):
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, n)]


# Hand-written stroke skeletons in a unit box (x right, y down). Several digits
# have more than one common way of being written.
SKELETONS = {
    0: [[_ellipse(.5, .5, .32, .5, -80, 285)]],
    1: [[[(.5, 0), (.5, 1)]],
        [[(.25, .22), (.5, 0), (.5, 1)]],
        [[(.25, .22), (.5, 0), (.5, 1)], [(.25, 1), (.75, 1)]],
        [[(.3, .15), (.55, 0), (.5, .5), (.45, 1)]]],
    2: [[[(.15, .25), (.3, .06), (.58, .0), (.82, .18), (.74, .45), (.15, 1), (.88, 1)]],
        [[(.12, .3), (.35, .02), (.75, .05), (.85, .3), (.6, .55), (.12, 1), (.9, .98)]],
        [[(.18, .2), (.5, 0), (.82, .2), (.5, .6), (.12, 1)], [(.12, 1), (.9, 1)]]],
    3: [[[(.15, .12), (.5, 0), (.8, .15), (.7, .4), (.42, .5), (.8, .6), (.86, .85), (.5, 1), (.12, .88)]],
        [[(.15, 0), (.85, 0), (.45, .45), (.8, .58), (.85, .85), (.5, 1), (.1, .88)]]],
    4: [[[(.65, 0), (.1, .65), (.92, .65)], [(.65, 0), (.65, 1)]],
        [[(.18, 0), (.15, .52), (.85, .52)], [(.72, 0), (.72, 1)]]],
    5: [[[(.82, 0), (.22, 0), (.15, .46), (.5, .4), (.8, .55), (.86, .8), (.5, 1), (.1, .9)]],
        [[(.2, .5), (.5, .4), (.8, .55), (.86, .8), (.5, 1), (.1, .9)], [(.2, .5), (.28, 0), (.82, 0)]]],
    6: [[[(.7, .04), (.32, .3), (.15, .65), (.3, .95), (.65, .96), (.86, .72), (.62, .5), (.2, .62)]],
        [[(.6, 0), (.2, .5), (.15, .8)] + _ellipse(.5, .72, .34, .27, 180, 540)[:-1]]],
    7: [[[(.1, 0), (.9, 0), (.4, 1)]],
        [[(.1, 0), (.9, 0), (.4, 1)], [(.3, .5), (.7, .5)]],
        [[(.12, .12), (.2, 0), (.88, 0), (.6, .45), (.4, 1)]]],
    8: [[_ellipse(.5, .27, .28, .25, 0, 360), _ellipse(.5, .74, .33, .27, 0, 360)],
        [[(.5, .5), (.2, .3), (.5, 0), (.8, .3), (.5, .5), (.15, .75), (.5, 1), (.85, .75), (.5, .5)]]],
    9: [[_ellipse(.5, .3, .3, .28, 0, 360), [(.8, .3), (.72, .7), (.45, 1)]],
        [[(.78, .3), (.5, .02), (.2, .3), (.5, .55), (.78, .3), (.72, .7), (.45, 1)]]],
}


def _densify(pts, step=5.0):
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n) for t in range(1, n + 1)]
    return out


def draw_digit(digit: int, rng: np.random.Generator, supersample: int = 3) -> np.ndarray:
    """Return a (280, 280) float32 canvas, white strokes on black, values 0-1."""
    strokes = SKELETONS[digit][rng.integers(len(SKELETONS[digit]))]

    size = rng.uniform(0.38, 0.92) * CANVAS            # digit height in px (small to huge)
    aspect = rng.uniform(0.6, 1.15)                    # width / height
    angle = math.radians(rng.normal(0, 8))
    shear = rng.uniform(-0.25, 0.25)
    cx = rng.uniform(size * aspect / 2 + 10, CANVAS - size * aspect / 2 - 10) if size * aspect < CANVAS - 20 else CANVAS / 2
    cy = rng.uniform(size / 2 + 10, CANVAS - size / 2 - 10) if size < CANVAS - 20 else CANVAS / 2
    wobble = rng.uniform(1.5, 5.0)                     # px of hand shake
    cos, sin = math.cos(angle), math.sin(angle)

    S = supersample  # supersample for smooth edges
    img = Image.new("L", (CANVAS * S, CANVAS * S), 0)
    dr = ImageDraw.Draw(img)
    for stroke in strokes:
        pts = []
        drift = np.zeros(2)
        for (ux, uy) in _densify([(p[0] * 1000, p[1] * 1000) for p in stroke], 12):
            ux, uy = ux / 1000, uy / 1000
            drift += rng.normal(0, wobble * 0.35, 2)    # slow wander + fast jitter
            drift *= 0.9
            x = (ux - .5) * size * aspect + shear * (uy - .5) * size
            y = (uy - .5) * size
            xr, yr = x * cos - y * sin, x * sin + y * cos
            pts.append(((cx + xr + drift[0] + rng.normal(0, wobble * .3)) * S,
                        (cy + yr + drift[1] + rng.normal(0, wobble * .3)) * S))
        r = LINE_WIDTH * S / 2
        dr.line(pts, fill=255, width=LINE_WIDTH * S, joint="curve")
        for (x, y) in (pts[0], pts[-1]):
            dr.ellipse([x - r, y - r, x + r, y + r], fill=255)
    img = img.resize((CANVAS, CANVAS), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _norm_worker(args):
    from preprocess import normalize_digit, normalize_pen_digit
    digit, n, seed, pen = args
    normalize = normalize_pen_digit if pen else normalize_digit
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = normalize(draw_digit(digit, rng, supersample=2))
        out.append(a if a is not None else np.zeros((28, 28), np.float32))
    return np.stack(out), np.full(n, digit)


def make_normalized_set(n_per_digit: int, seed: int = 0, workers: int = 2, pen: bool = False):
    """Simulated drawings already passed through normalize_digit() -> (N,28,28), labels (N,)."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(d, n_per_digit // 4, seed * 7919 + d * 31 + k, pen) for d in range(10) for k in range(4)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_norm_worker, jobs))
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def make_set(n_per_digit: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for d in range(10):
        for _ in range(n_per_digit):
            X.append(draw_digit(d, rng)); y.append(d)
    return np.stack(X), np.array(y)
