# src/dataset.py

import csv
import os
import numpy as np

LANDMARK_COUNT = 21


def ensure_csv(csv_path: str):
    """Create CSV file with header if it doesn't exist."""
    if os.path.isfile(csv_path):
        return

    header = ["sign"]
    for i in range(LANDMARK_COUNT):
        header.extend([f"{i}x", f"{i}y"])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def save_sample(csv_path: str, label: str, landmarks: list[int]):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([label] + landmarks)


def load_dataset(csv_path: str):
    data, labels = [], []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            labels.append(row[0])
            data.append([int(v) for v in row[1:]])

    return np.array(data), np.array(labels)
