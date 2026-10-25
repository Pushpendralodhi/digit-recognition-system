"""
src/build_pen_training_set.py
Builds the training set the CNN is trained on (see README, "How drawing reliability was fixed"):

  A  raw MNIST                                  - the original 28x28 digits
  B  MNIST -> distorted -> pen-normalised       - the same real handwriting, in the format the app feeds the model
  C  simulated mouse-drawn digits -> pen-norm.  - many handwriting styles, any size/position (canvas_sim.py)
  D  stroke-font digits -> pen-normalised       - 4 Hershey fonts (3 other fonts are held out for testing)

Output: <out>/pen_train.npz with uint8 images (N,28,28) and int labels - a few hundred MB, so it
is NOT kept inside the project; regenerate it any time with:
    python src/build_pen_training_set.py --out work/
"""
import argparse, os, time
import numpy as np
from data_loader import load_mnist
from augment import pen_dataset
from canvas_sim import make_normalized_set
from font_sim import make_font_set, TRAIN_FONTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work")
    ap.add_argument("--distorted-copies", type=int, default=2)
    ap.add_argument("--sim-per-digit", type=int, default=3000)
    ap.add_argument("--font-per-digit", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    d = load_mnist()
    Xtr = d["X_train_cnn"].reshape(-1, 28, 28); ytr = d["y_train_int"]
    Xva = d["X_val_cnn"].reshape(-1, 28, 28);   yva = d["y_val_int"]
    onehot_free = lambda v: np.asarray(v, dtype=np.int8)

    train_X, train_y, val_X, val_y = [Xtr], [ytr], [Xva], [yva]            # A
    log(f"A raw MNIST: {len(Xtr)} train / {len(Xva)} val")

    Xb = pen_dataset(Xtr, copies=1, seed=1, workers=a.workers, distort=False)   # B (clean)
    train_X.append(Xb); train_y.append(ytr); log("B clean pen-normalised MNIST done")
    Xb2 = pen_dataset(Xtr, copies=a.distorted_copies, seed=2, workers=a.workers, distort=True)   # B (distorted)
    train_X.append(Xb2); train_y.append(np.tile(ytr, a.distorted_copies)); log("B distorted pen-normalised MNIST done")
    val_X.append(pen_dataset(Xva, copies=1, seed=3, workers=a.workers, distort=False)); val_y.append(yva)

    Xs, ys = make_normalized_set(a.sim_per_digit, seed=11, workers=a.workers, pen=True)               # C
    train_X.append(Xs); train_y.append(ys)
    Xsv, ysv = make_normalized_set(max(40, a.sim_per_digit // 10), seed=12, workers=a.workers, pen=True)
    val_X.append(Xsv); val_y.append(ysv); log(f"C simulated drawings done ({len(Xs)} train)")

    Xf, yf = make_font_set(a.font_per_digit, fonts=TRAIN_FONTS, seed=21, workers=a.workers)            # D
    train_X.append(Xf); train_y.append(yf)
    Xfv, yfv = make_font_set(max(40, a.font_per_digit // 10), fonts=TRAIN_FONTS, seed=22, workers=a.workers)
    val_X.append(Xfv); val_y.append(yfv); log(f"D font digits done ({len(Xf)} train)")

    to8 = lambda arrs: (np.clip(np.concatenate(arrs), 0, 1) * 255 + 0.5).astype(np.uint8)
    np.savez_compressed(os.path.join(a.out, "pen_train.npz"),
                        X_train=to8(train_X), y_train=np.concatenate(train_y).astype(np.int8),
                        X_val=to8(val_X), y_val=np.concatenate(val_y).astype(np.int8))
    log(f"saved: train {sum(len(x) for x in train_X):,}  val {sum(len(x) for x in val_X):,}")


if __name__ == "__main__":
    main()
