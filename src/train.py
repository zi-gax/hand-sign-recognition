# src/train.py

import os
import joblib
from sklearn.neighbors import KNeighborsClassifier

from config import CSV_PATH, MODEL_PATH, KNN_NEIGHBORS
from dataset import load_dataset


def train_and_save():
    X, y = load_dataset(CSV_PATH)

    if len(X) == 0:
        raise RuntimeError("Dataset is empty. Collect data first.")

    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"[✓] Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
