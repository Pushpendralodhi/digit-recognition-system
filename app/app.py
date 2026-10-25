"""
app/app.py
Flask web app: draw a digit on a canvas, get a real-time prediction.

Run with:
    cd digit-recognition-system
    python app/app.py
Then open http://127.0.0.1:5000
"""

import base64
import io
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # allow "from predictor import ..."
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))  # shared preprocessing

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

from predictor import get_predictor
from preprocess import normalize_pen_digit

app = Flask(__name__)


@app.route("/")
def index():
    predictor = get_predictor()
    return render_template("index.html", backend=predictor.backend)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    data_url = payload.get("image", "")
    if "," not in data_url:
        return jsonify({"error": "No image data received."}), 400

    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    # The canvas is a 280x280 white-on-black pen drawing. Don't just squash it to 28x28:
    # thin it to a fixed-width pen line, then crop/scale/centre it the way MNIST digits are
    # (preprocess.normalize_pen_digit - the browser app does exactly the same steps).
    canvas = np.array(Image.open(io.BytesIO(img_bytes)).convert("L")).astype("float32") / 255.0
    arr = normalize_pen_digit(canvas)
    if arr is None:
        return jsonify({"error": "Nothing drawn yet."}), 400

    predictor = get_predictor()
    pred, probs = predictor.predict(arr)

    return jsonify({
        "prediction": pred,
        "probabilities": probs,
        "backend": predictor.backend,
    })


if __name__ == "__main__":
    print("Starting Handwritten Digit Recognition app on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
