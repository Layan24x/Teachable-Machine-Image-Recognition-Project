import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("models/keras_model.h5")
LABELS_PATH = Path("models/labels.txt")
IMAGE_SIZE = (224, 224)


def load_labels(path: Path):
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Teachable Machine commonly exports labels as: "0 ClassName"
            parts = line.split(maxsplit=1)
            labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return labels


def predict(image_path: Path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Export your Teachable Machine model "
            "as TensorFlow/Keras and place keras_model.h5 in the models folder."
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_PATH}. Place labels.txt in the models folder."
        )

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = load_labels(LABELS_PATH)

    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image).astype(np.float32)
    normalized = (image_array / 127.5) - 1.0
    data = np.expand_dims(normalized, axis=0)

    prediction = model.predict(data, verbose=0)[0]
    index = int(np.argmax(prediction))

    if index >= len(labels):
        label = f"Class {index}"
    else:
        label = labels[index]

    confidence = float(prediction[index]) * 100

    print("=" * 50)
    print("Teachable Machine Image Classification")
    print("=" * 50)
    print(f"Input image : {image_path}")
    print(f"Predicted class: {label}")
    print(f"Confidence    : {confidence:.2f}%")
    print("=" * 50)

    print("\nAll class probabilities:")
    for i, score in enumerate(prediction):
        name = labels[i] if i < len(labels) else f"Class {i}"
        print(f"- {name}: {float(score) * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a Teachable Machine Keras model and classify an input image."
    )
    parser.add_argument("image", help="Path to the input image, e.g. test_images/cat.jpg")
    args = parser.parse_args()

    predict(Path(args.image))
