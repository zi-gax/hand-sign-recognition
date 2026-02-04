# src/hand_tracking.py

import cv2
import mediapipe as mp

from config import *
from dataset import ensure_csv, save_sample


def get_hand_landmarks(hand_landmarks, width, height):
    """Return landmarks relative to landmark 0."""
    base = hand_landmarks.landmark[0]
    base_x, base_y = base.x * width, base.y * height

    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(int(lm.x * width - base_x))
        coords.append(int(lm.y * height - base_y))
    return coords


def mirror_landmarks(landmarks: list[int]):
    mirrored = []
    for i in range(0, len(landmarks), 2):
        mirrored.extend([-landmarks[i], landmarks[i + 1]])
    return mirrored


def main():
    ensure_csv(CSV_PATH)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF
    )
    drawer = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAMERA_INDEX)
    last_landmarks = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            last_landmarks = get_hand_landmarks(hand, w, h)
            drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking (S = save, ESC = quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and last_landmarks:
            hand_type = input("Hand (L/R): ").strip().upper()
            name = input("Sign name: ").strip()

            original = last_landmarks
            mirrored = mirror_landmarks(original)

            if hand_type == "R":
                save_sample(CSV_PATH, name + "R", original)
                save_sample(CSV_PATH, name + "L", mirrored)
            else:
                save_sample(CSV_PATH, name + "L", original)
                save_sample(CSV_PATH, name + "R", mirrored)

            print(f"[✓] Saved sign '{name}'")

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
