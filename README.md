# ✋ Hand Sign Recognition (Real-Time)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)

A **real-time hand sign recognition system** built with **MediaPipe**, **OpenCV**, and **scikit-learn**.  
Supports left/right hand normalization, dataset augmentation through mirroring, and fast inference using a pre-trained KNN model.

Designed with **clean architecture**, **modular code**, and **production-ready practices**.

---

## 🚀 Features

- Real-time hand landmark detection (MediaPipe Hands)  
- Relative landmark normalization (wrist-based)  
- Automatic left/right hand mirroring for dataset augmentation  
- KNN classifier with probability output  
- Temporal prediction smoothing (voting window)  
- Model saving/loading using `joblib`  

---

## 📁 Project Structure

```
hand-sign-recognition/
├── src/
│   ├── config.py            # Configuration constants
│   ├── dataset.py           # CSV handling & dataset utilities
│   ├── train.py             # Train & save KNN model
│   ├── hand_tracking.py     # Collect landmarks from camera
│   └── predict_realtime.py  # Realtime prediction using saved model
├── data/
│   └── hand_sign_dataset.csv
├── models/
│   └── knn_model.joblib
├── assets/
│   ├── handL.png
│   └── handR.png
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/zi-gax/hand-sign-recognition.git
cd hand-sign-recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📸 Dataset Collection

Run the hand tracking script to collect training data:

```bash
python src/hand_tracking.py
```

**Controls**:

| Key / Action  | Description |
|---------------|-------------|
| `S`           | Save current hand landmarks |
| `L` / `R`     | Choose hand type |
| Input name    | Sign name to store |
| `ESC`         | Quit |

> Mirrored samples are generated automatically.  
> Landmarks are **relative to the wrist**, making the dataset scale-invariant.

---

## 🏋️ Train Model

Train KNN model and save to disk:

```bash
python src/train.py
```

This generates:  

```
models/knn_model.joblib
```

---

## 🤖 Real-Time Prediction

Run real-time inference using the saved model:

```bash
python src/predict_realtime.py
```

**Display**:

- Final stabilized prediction  
- Top 3 predicted classes with probability  
- Live hand skeleton overlay  

> If no model is found, the system automatically trains one.

---

## 📝 Configuration

All configurable parameters are in:

```
src/config.py
```

Includes:

- Camera index  
- MediaPipe detection/tracking confidence  
- KNN neighbors  
- Prediction smoothing history  
- File paths  

---

## ⭐ Acknowledgments

- [Google MediaPipe](https://developers.google.com/mediapipe)  
- OpenCV community  
- scikit-learn contributors
