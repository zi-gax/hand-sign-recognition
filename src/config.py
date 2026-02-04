# src/config.py

CAMERA_INDEX = 1

# Paths
CSV_PATH = "../data/hand_sign_dataset.csv"
MODEL_PATH = "../models/knn_model.joblib"

# MediaPipe
MAX_HANDS = 1
MODEL_COMPLEXITY = 1
DETECTION_CONF = 0.7
TRACKING_CONF = 0.7

# KNN
KNN_NEIGHBORS = 3
PREDICTION_HISTORY = 5
