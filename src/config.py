# src/config.py

CAMERA_INDEX = 0

# Paths
from pathlib import Path

# Get the directory where this script resides
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if main.py is in /src

# Define paths relative to BASE_DIR
CSV_PATH = BASE_DIR / "data" / "hand_sign_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "knn_model.joblib"

# MediaPipe
MAX_HANDS = 1
MODEL_COMPLEXITY = 1
DETECTION_CONF = 0.7
TRACKING_CONF = 0.7

# KNN
KNN_NEIGHBORS = 3
PREDICTION_HISTORY = 5
