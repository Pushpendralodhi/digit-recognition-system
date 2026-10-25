"""
src/models.py
Builds the two model architectures used throughout this project.
Keeping them here (instead of copy-pasted in every script) means
train.py, evaluate.py and the Flask app all use the exact same
architecture definition.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten


def build_ann(input_dim: int = 784, num_classes: int = 10) -> Sequential:
    """Multi-Layer Perceptron: 784 -> 128 -> 64 -> 10"""
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ], name="ANN")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_shape=(28, 28, 1), num_classes: int = 10) -> Sequential:
    """CNN: Conv(32) -> Pool -> Conv(64) -> Pool -> Dense(128) -> 10"""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ], name="CNN")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model(kind: str):
    """kind: 'ann' or 'cnn'"""
    if kind == "ann":
        return build_ann()
    elif kind == "cnn":
        return build_cnn()
    raise ValueError(f"Unknown model kind: {kind!r} (expected 'ann' or 'cnn')")


if __name__ == "__main__":
    build_ann().summary()
    build_cnn().summary()
