"""
notebooks/colab_full_pipeline.py
One self-contained script covering the whole ML pipeline on the REAL MNIST
dataset: load -> preprocess -> build ANN + CNN -> train both -> evaluate ->
compare -> save. Use this if you just want one file to paste into Google
Colab rather than uploading the whole project structure.

Google Colab: https://colab.research.google.com -> New notebook -> paste
this whole file into a cell -> Shift+Enter. Turn on a GPU (Runtime > Change
runtime type > T4 GPU) for faster training.

Downloads at the end: ann_model.keras, cnn_model.keras, and all the
charts — download them (folder icon on the left in Colab) and drop the
.keras files into saved_models/ in the project so the Flask app picks up
the real trained models automatically.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# =====================================================================
# 1. LOAD + PREPROCESS
# =====================================================================
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

X_train_ann_full = X_train.reshape(-1, 784)
X_test_ann = X_test.reshape(-1, 784)
X_train_cnn_full = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

idx_tr, idx_val = train_test_split(np.arange(len(X_train)), test_size=0.1, random_state=42, stratify=y_train)

Xtr_ann, Xval_ann = X_train_ann_full[idx_tr], X_train_ann_full[idx_val]
Xtr_cnn, Xval_cnn = X_train_cnn_full[idx_tr], X_train_cnn_full[idx_val]
ytr, yval = y_train_cat[idx_tr], y_train_cat[idx_val]

print(f"Train: {len(idx_tr)}  Val: {len(idx_val)}  Test: {len(X_test)}")

# =====================================================================
# 2. BUILD MODELS
# =====================================================================
def build_ann():
    m = Sequential([
        Dense(128, activation="relu", input_shape=(784,)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ], name="ANN")
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def build_cnn():
    m = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax"),
    ], name="CNN")
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

ann = build_ann(); ann.summary()
cnn = build_cnn(); cnn.summary()

# =====================================================================
# 3. TRAIN (early stopping + best-checkpoint saving)
# =====================================================================
EPOCHS, BATCH = 25, 128

print("\n===== Training ANN =====")
ann_hist = ann.fit(
    Xtr_ann, ytr, validation_data=(Xval_ann, yval), epochs=EPOCHS, batch_size=BATCH,
    callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
               ModelCheckpoint("ann_model.keras", monitor="val_accuracy", save_best_only=True)],
    verbose=1,
)

print("\n===== Training CNN =====")
cnn_hist = cnn.fit(
    Xtr_cnn, ytr, validation_data=(Xval_cnn, yval), epochs=EPOCHS, batch_size=BATCH,
    callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
               ModelCheckpoint("cnn_model.keras", monitor="val_accuracy", save_best_only=True)],
    verbose=1,
)

# =====================================================================
# 4. MONITOR: accuracy / loss curves
# =====================================================================
def plot_history(history, name):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{name} Accuracy"); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(f"{name.lower()}_accuracy.png", dpi=150, bbox_inches="tight"); plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{name} Loss"); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(f"{name.lower()}_loss.png", dpi=150, bbox_inches="tight"); plt.show()

plot_history(ann_hist, "ANN")
plot_history(cnn_hist, "CNN")

# =====================================================================
# 5. EVALUATE: reload best checkpoints, test-set metrics + confusion matrices
# =====================================================================
ann_best = load_model("ann_model.keras")
cnn_best = load_model("cnn_model.keras")

def evaluate(model, X_test_, name):
    probs = model.predict(X_test_, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = (preds == y_test).mean()
    print(f"\n===== {name} test results =====")
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=4))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} Confusion Matrix")
    plt.savefig(f"{name.lower()}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    return acc

ann_acc = evaluate(ann_best, X_test_ann, "ANN")
cnn_acc = evaluate(cnn_best, X_test_cnn, "CNN")

# =====================================================================
# 6. COMPARE
# =====================================================================
plt.figure(figsize=(6, 5))
bars = plt.bar(["ANN", "CNN"], [ann_acc, cnn_acc], color=["#2c6e9e", "#a83232"])
plt.ylabel("Test Accuracy"); plt.ylim(0, 1); plt.title("ANN vs CNN — Test Accuracy")
for b, a in zip(bars, [ann_acc, cnn_acc]):
    plt.text(b.get_x() + b.get_width()/2, a + 0.02, f"{a:.4f}", ha="center", fontweight="bold")
plt.savefig("ann_vs_cnn_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nFinal: ANN test accuracy = {ann_acc:.4f} | CNN test accuracy = {cnn_acc:.4f}")
print("Download ann_model.keras and cnn_model.keras and place them in saved_models/")
print("so the Flask app (app/app.py) uses the real trained models.")
