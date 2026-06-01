# 🚗 Driver Drowsiness Detection

Real-time driver drowsiness detection system using a custom CNN model with MediaPipe face mesh and OpenCV.

## Overview

This project detects driver drowsiness in real-time using a combination of:
- **Eye Aspect Ratio (EAR)** — detects eye closure
- **Mouth Aspect Ratio (MAR)** — detects yawning
- **CNN Model** — classifies face crops as Drowsy / Non-Drowsy

A composite fatigue score triggers visual and audio alerts when the driver appears drowsy.

## Project Structure

```
├── drowsiness_model.keras          # Trained Keras model (~7.6 MB)
├── model/
│   ├── drowsiness_model_final.keras  # Final trained model (~5.1 MB)
│   └── drowsiness_model.tflite       # TFLite model for mobile/edge deployment
├── colab_training.py               # Training script (run on Google Colab)
├── colab_optimized.py              # Optimized training variant
├── realtime_detection.py           # Real-time detection script (run locally)
└── README.md
```

## Training

The model was trained on the [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) (~41,790 images) using Google Colab.

### Architecture
- Custom CNN with 4 convolutional blocks + BatchNorm + Dropout
- Input size: 128×128 RGB
- Binary classification (Drowsy / Non-Drowsy)
- Optional MobileNetV2 transfer learning variant

### To retrain
1. Open `colab_training.py` in Google Colab
2. Run each cell sequentially
3. Download the trained model files

## Real-Time Detection

### Prerequisites
```bash
pip install opencv-python mediapipe numpy tensorflow pygame
```

### Run
```bash
python realtime_detection.py
```

### Features
- Live webcam feed with face mesh overlay
- HUD dashboard showing EAR, MAR, fatigue score, and FPS
- Color-coded status bar (ALERT / SLIGHTLY DROWSY / DROWSY)
- Audio alarm when drowsiness is detected
- Press **Q** to quit

## Tech Stack

- **TensorFlow / Keras** — model training and inference
- **MediaPipe** — face mesh landmark detection
- **OpenCV** — video capture and visualization
- **NumPy** — numerical computations
- **Pygame** — audio alerts

## License

This project is for educational purposes.
