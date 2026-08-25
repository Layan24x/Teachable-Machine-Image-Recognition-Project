from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("models/keras_model.h5")
LABELS_PATH = Path("models/labels.txt")
TEST_DIR = Path("test_images")
IMAGE_SIZE = (224, 224)


def load_labels(path):
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return labels


def main():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = load_labels(LABELS_PATH)

    total = 0
    correct = 0

    print("Evaluation")
    print("-" * 50)

    for class_dir in sorted(p for p in TEST_DIR.iterdir() if p.is_dir()):
        expected = class_dir.name
        class_total = 0
        class_correct = 0

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
            arr = np.asarray(image).astype(np.float32)
            arr = (arr / 127.5) - 1.0
            prediction = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            predicted = labels[predicted_index]

            class_total += 1
            total += 1

            if predicted.lower() == expected.lower():
                class_correct += 1
                correct += 1

        accuracy = (class_correct / class_total * 100) if class_total else 0
        print(f"{expected}: {class_correct}/{class_total} correct ({accuracy:.2f}%)")

    overall = (correct / total * 100) if total else 0
    print("-" * 50)
    print(f"Overall accuracy: {correct}/{total} = {overall:.2f}%")


if __name__ == "__main__":
    main()
