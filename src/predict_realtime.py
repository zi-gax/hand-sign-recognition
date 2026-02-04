# src/predict_realtime.py

import os
import cv2
import joblib
import mediapipe as mp
import numpy as np
from collections import deque

from config import *
from dataset import load_dataset  # optional safety


def extract_landmarks(hand_landmarks, w, h):
    base = hand_landmarks.landmark[0]
    bx, by = base.x * w, base.y * h

    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(int(lm.x * w - bx))
        coords.append(int(lm.y * h - by))
    return coords


def load_or_train_model():
    if os.path.isfile(MODEL_PATH):
        print("[✓] Loading saved model...")
        return joblib.load(MODEL_PATH)

    print("[!] Model not found. Training new model...")
    from train import train_and_save
    train_and_save()
    return joblib.load(MODEL_PATH)


def main():
    model = load_or_train_model()
    history = deque(maxlen=PREDICTION_HISTORY)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF
    )
    drawer = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            sample = extract_landmarks(hand, w, h)
            probs = model.predict_proba([sample])[0]
            classes = model.classes_

            top_idx = np.argsort(probs)[-3:][::-1]
            history.append(classes[top_idx[0]])
            final = max(set(history), key=history.count)

            cv2.putText(frame, f"Sign: {final}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            for i, idx in enumerate(top_idx):
                cv2.putText(frame,
                            f"{i+1}. {classes[idx]} {probs[idx]*100:.1f}%",
                            (w - 220, 40 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0) if i == 0 else (200, 200, 200),
                            1)

        cv2.imshow("Hand Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
