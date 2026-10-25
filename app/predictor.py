"""
app/predictor.py
A single predict() interface used by the Flask app, regardless of which
backend is actually available:

  1. Preferred: the real Keras CNN or ANN trained on full 28x28 MNIST
     (src/train.py output, saved_models/cnn_model.keras or ann_model.keras).
  2. Fallback: a scikit-learn MLP trained on the offline 8x8 digits dataset
     (saved_models/ann_sklearn_demo.joblib), used automatically if no Keras
     model file is found — e.g. on a machine without TensorFlow/internet.
     This keeps the app runnable end-to-end for a demo even before you've
     trained the real model.

Swap in the real model at any time by running src/train.py and dropping
the resulting .keras file into saved_models/ — no app code changes needed.
"""

import os
import numpy as np
from PIL import Image

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


class Predictor:
    def __init__(self):
        self.backend = None
        self.model = None
        self._load_best_available()

    def _load_best_available(self):
        # 1. Try the real Keras CNN, then Keras ANN
        for kind in ("cnn", "ann"):
            path = os.path.join(SAVED_MODELS_DIR, f"{kind}_model.keras")
            if os.path.exists(path):
                try:
                    from tensorflow.keras.models import load_model
                    self.model = load_model(path)
                    self.backend = f"keras-{kind}"
                    print(f"[predictor] Loaded real Keras {kind.upper()} model from {path}")
                    return
                except Exception as e:
                    print(f"[predictor] Found {path} but couldn't load it ({e}); trying next option.")

        # 2. Fall back to the offline scikit-learn demo model
        fallback_path = os.path.join(SAVED_MODELS_DIR, "ann_sklearn_demo.joblib")
        if os.path.exists(fallback_path):
            import joblib
            self.model = joblib.load(fallback_path)
            self.backend = "sklearn-demo"
            print(f"[predictor] No Keras model found — using offline demo model from {fallback_path}")
            return

        raise FileNotFoundError(
            "No model found. Either run src/train.py to train the real Keras model, "
            "or run scripts/train_offline_fallback.py to create the offline demo model."
        )

    def predict(self, image_28x28: np.ndarray):
        """
        image_28x28: float array, shape (28, 28), values 0-1, digit as
        white strokes on a black background (MNIST convention).
        Returns (predicted_digit: int, probabilities: list[float] len 10)
        """
        if self.backend.startswith("keras-"):
            kind = self.backend.split("-")[1]
            x = image_28x28.reshape(1, 784) if kind == "ann" else image_28x28.reshape(1, 28, 28, 1)
            probs = self.model.predict(x, verbose=0)[0]
        else:
            # sklearn demo model was trained on 8x8 images scaled 0-16
            small = np.array(Image.fromarray((image_28x28 * 255).astype("uint8")).resize((8, 8)))
            x = (small.astype("float32") / 255.0).reshape(1, -1)
            probs = self.model.predict_proba(x)[0]

        pred = int(np.argmax(probs))
        return pred, [float(p) for p in probs]


_predictor_singleton = None


def get_predictor() -> Predictor:
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = Predictor()
    return _predictor_singleton
