"""
scripts/export_web_weights.py
Exports the trained CNN's weights to the flat files the static web app
(webapp_static/) loads: weights_b64.json (all float32 weights, base64) and
manifest.json (layer list with each tensor's shape/offset/count).

The browser runs a small hand-written forward pass, so the architecture must
stay Conv32 -> Pool -> Conv64 -> Pool -> Flatten -> Dense128 -> Dense10.

Usage:
    python scripts/export_web_weights.py --model saved_models/cnn_model.keras --out webapp_static
"""
import argparse, base64, json, os
import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saved_models/cnn_model.keras")
    ap.add_argument("--out", default="webapp_static")
    a = ap.parse_args()

    model = tf.keras.models.load_model(a.model)
    flat, manifest, offset = [], [], 0
    for layer in model.layers:
        entry = {"type": type(layer).__name__, "name": layer.name}
        ws = layer.get_weights()
        if ws:
            entry["weights"] = []
            for w in ws:
                w = np.asarray(w, dtype="<f4")
                entry["weights"].append({"shape": list(w.shape), "offset": offset, "count": int(w.size)})
                flat.append(w.ravel()); offset += int(w.size)
        manifest.append(entry)

    blob = np.concatenate(flat).astype("<f4").tobytes()
    os.makedirs(a.out, exist_ok=True)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    json.dump({"data": base64.b64encode(blob).decode()}, open(os.path.join(a.out, "weights_b64.json"), "w"))
    print(f"Exported {offset:,} weights -> {a.out}/weights_b64.json + manifest.json")


if __name__ == "__main__":
    main()
