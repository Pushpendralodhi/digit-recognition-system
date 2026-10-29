# Handwritten Digit Recognition System

A complete, working implementation of the MNIST handwritten digit recognizer:
real dataset, ANN, CNN, evaluation, and a live web app with a drawing
canvas — organized as a real project rather than one-off milestone scripts.

**This is not a demo.** The real MNIST dataset is included
(`data/raw/`, verified against the official MD5 checksums), and both
models are already trained on it and committed in `saved_models/`:

| Model | Test accuracy (10,000 real MNIST test images) |
|---|---|
| ANN (Dense 128 → 64) | **97.93%** |
| CNN (Conv32 → Conv64) | **99.41%** |

The web app loads the CNN automatically and serves real predictions —
clone it, install dependencies, and run it; no training step required.

## Live demo (already deployed, nothing to install)

**https://claude.ai/artifact/1hySXWczGWtRsYN5n7Ecu2**

This is the Flask app's exact UI, rebuilt as one static page that runs the
real trained CNN's weights directly in the browser (a small hand-written
JavaScript forward pass — no TensorFlow.js, no server). It includes a
"Try a real MNIST test digit" button so you can check real predictions
against real test-set labels immediately. `webapp_static/` in this project
is the source for that page, in case you want to host it yourself (e.g.
on GitHub Pages) instead of the Flask/gunicorn version.

## Project structure

```
digit-recognition-system/
├── requirements.txt
├── data/
│   └── raw/                # the 4 original MNIST idx-ubyte.gz files (verified MD5)
├── webapp_static/           # client-side-only version (real CNN weights, pure JS, no server) — this is what's live at the link above
├── src/
│   ├── preprocess.py       # pen normalisation (thin strokes to a fixed-width pen) + MNIST-style crop / scale / centre for drawn digits
│   ├── augment.py          # random tilt / slant / stretch / shaky-stroke / thickness distortions for training
│   ├── canvas_sim.py       # simulates mouse & touchpad drawings (many handwriting styles) for training/test
│   ├── font_sim.py         # stroke-font digits: 4 fonts for training, 3 held out for testing
│   ├── build_pen_training_set.py  # builds the 253k-image training set the CNN is trained oning
│   ├── data_loader.py      # loads & preprocesses MNIST from data/raw/ (no internet needed)
│   ├── models.py           # build_ann(), build_cnn() — single source of truth
│   ├── train.py            # CLI: train either model, early stopping, checkpointing
│   ├── evaluate.py         # CLI: accuracy, classification report, confusion matrix
│   └── predict.py          # CLI: predict the digit in a single image file
├── app/
│   ├── app.py               # Flask server
│   ├── predictor.py         # loads the best available model (real CNN first)
│   └── templates/index.html # canvas UI — draw a digit, get a live prediction
├── results/                 # real accuracy/loss curves, confusion matrices, comparison chart
├── saved_models/
│   ├── ann_model.keras      # trained, 97.93% test accuracy
│   └── cnn_model.keras      # trained, 99.41% test accuracy — used by the app
├── wsgi.py                  # production entry point (gunicorn)
├── Procfile                 # deploy command for Render/Railway/Heroku-style hosts
└── notebooks/
    └── colab_full_pipeline.py   # single-file version of the whole pipeline, for Colab/retraining
```

## Quick start — run the real thing

```bash
pip install -r requirements.txt
python app/app.py
```

Open http://127.0.0.1:5000, draw a digit, click Predict. The badge at the
top of the page reads **"model: keras-cnn"**, confirming the real trained
CNN (99.41% test accuracy) is serving the prediction — not a stand-in.

## Running in production (gunicorn)

```bash
pip install -r requirements.txt
gunicorn wsgi:application --bind 0.0.0.0:8000
```

This is what `Procfile` tells a hosting platform to run. See
"Deploying on GitHub Pages" below.

## Retraining from scratch

The dataset is already included, so this works offline, no Colab needed (a
CPU-only machine is fine). The first run builds the "pen" training set
(253,000 images, ~25 min on 2 cores, then cached in `work/`), then trains the CNN
(~45 s per epoch, about 10 epochs):

```bash
python src/build_pen_training_set.py --out work/     # optional - train.py does it automatically
python src/train.py --model cnn --epochs 14
python src/train.py --model ann --epochs 20
python src/evaluate.py --compare
python scripts/export_web_weights.py                 # refresh webapp_static/ with the new CNN weights
python scripts/robustness_check.py --model saved_models/cnn_model.keras
```

This overwrites `saved_models/*.keras` and regenerates the charts in
`results/`. `notebooks/colab_full_pipeline.py` is still provided as a
single-file version if you'd rather run/retrain on Google Colab.

## Why drawn digits used to fail, and how it was fixed

MNIST digits are slim pen lines, cropped to the digit, scaled so the longer side
is 20 px and centred by centre of mass. A canvas drawing is different in two ways:

1. **Size and position** - squashing the whole 280x280 canvas to 28x28 made small,
   large or off-centre digits unrecognisable. Fixed by cropping/scaling/centring
   exactly like MNIST (`src/preprocess.py: normalize_digit`).
2. **Pen thickness** - the canvas pen is a constant 18 px, so a small digit looks
   fat and its loops fill in: a 6 became a blob the model read as 0. Fixed by
   **pen normalisation** (`preprocess.normalize_pen_digit`): scale the drawing to a
   fixed size, thin every stroke to a 1-pixel centre line (Zhang-Suen thinning),
   redraw it with one fixed MNIST-like pen, *then* crop/scale/centre. Pen size and
   digit size stop mattering. `webapp_static/index.html` implements the same steps
   in JavaScript and produces the identical 28x28 image (verified pixel-for-pixel
   against the Python version on 70 drawings), and the Flask app uses the Python one.

The CNN was then retrained on data in that same format
(`src/build_pen_training_set.py`, 253,000 images): raw MNIST; MNIST distorted
(tilt, slant, stretch, shaky hand) and re-drawn through the pen pipeline;
simulated mouse drawings (`src/canvas_sim.py`); and stroke-font digits from 4
Hershey fonts (`src/font_sim.py`). Three other fonts are never trained on.

Measured with `python scripts/robustness_check.py` (accuracy):

| Test set | squash (original) | pen pipeline (now) |
|---|---|---|
| Clean MNIST test set, 10,000 images | 99.05% (old model) | **99.41%** |
| Held-out stroke fonts, 1,000 drawings (fonts never trained on) | 57.1% | **100.0%** |
| Simulated mouse drawings, 1,000 (same generator as training - easiest set) | 73.4% | **99.9%** |
| Typeset TTF digits, 600 (thick filled glyphs, not pen strokes) | 23.5% | 92.4% |
| Real web page driven by simulated mouse movements, 225 drawings | - | **225 / 225** |

Honest limits: these are synthetic drawings (fonts and a stroke generator), not
hundreds of real people - nothing here can promise 100% on everyone's handwriting.
Typeset/decorative fonts are the weakest case (92%), and very unusual handwriting can
still be misread; try the app yourself on the styles you write.

## Predicting a single image from the command line

```bash
python src/predict.py --model cnn --image path/to/digit.png
```

Works with any photo of a handwritten digit — it's auto-converted to
grayscale, resized, and inverted if needed to match MNIST's white-on-black
convention.

## Deploying on GitHub Pages (free, recommended)

The website (`index.html`) is already at the top level of this repo and also in `docs/`, so in Settings -> Pages choose Source: Deploy from a branch -> `main` and either `/ (root)` or `/docs` -> Save. Both work.

`webapp_static/` is a complete static website. Create a free GitHub account, make a
new public repository, upload the files **inside** `webapp_static/` (Add file →
Upload files), then Settings → Pages → Deploy from a branch → `main` / `(root)` →
Save. After a minute or two it's live at `https://<username>.github.io/<repo>/`.

## Deploying the Flask version

**Already deployed** — see "Live demo" above; that's a genuinely public
URL with the real trained CNN, nothing further to do.

If you specifically want the Flask/Python version (e.g. for a server-side
project writeup, or because you want `/predict` as a real HTTP API) it is
deployment-ready too (`wsgi.py` + `Procfile` + `gunicorn` in
`requirements.txt`), and was verified under gunicorn in development.
To get it onto its own public URL, push this project to a free host that
runs Python/Flask apps — for example:

- **Render** (render.com) — new Web Service → connect/upload this repo →
  build command `pip install -r requirements.txt`, start command
  `gunicorn wsgi:application`.
- **Railway** (railway.app) — new project → deploy from repo/folder; it
  auto-detects the `Procfile`.
- **PythonAnywhere** — upload the project, create a new web app, point it
  at `wsgi.py`.

All three have a free tier and need no code changes — just an account on
that platform, which only you can create.

## Why two models

The ANN (flattened pixels through dense layers) is faster to train and a
reasonable baseline (97.93%). The CNN (convolution + pooling) learns
spatial patterns like strokes and curves directly, which is why it scores
higher on image data (99.41%) — see `results/ann_vs_cnn_comparison.png`
and `results/*_confusion_matrix.png` for the real, measured comparison.

## Notes

- The web app's UI is original — not a copy of any reference repository's
  styling — dark navy/amber theme with a canvas that mimics MNIST's
  white-on-black convention so predictions are more accurate.
- `predictor.py` is the one place that decides which model backend to
  use; everything else (the Flask routes, the HTML) is unaffected by
  which one is active. It still falls back to the ANN, then to a small
  scikit-learn model, purely as a safety net if a `.keras` file is ever
  missing — but with both real models present, it always uses the CNN.
- `data/raw/*.gz` are the original, unmodified MNIST files (verified by
  MD5 against the published checksums) — see `data/raw/SOURCE.md`.

## Installing it as an app (PWA)

`webapp_static/` is an installable Progressive Web App (manifest, icons and a
service worker are included). Host that folder over HTTPS (e.g. GitHub Pages),
open the URL on a phone or laptop, then: Chrome/Android -> menu -> "Install app" /
"Add to Home screen"; Safari/iPhone -> Share -> "Add to Home Screen"; desktop
Chrome/Edge -> install icon in the address bar. After the first load it works
fully offline.
