# =============================================================================
#  REAL-TIME DRIVER DROWSINESS DETECTION
#  Run this on your LOCAL machine after training on Colab.
#  Uses: OpenCV + MediaPipe + Your Trained Model
# =============================================================================
#
#  Setup (run once):
#    pip install opencv-python mediapipe numpy tensorflow pygame
#
#  Usage:
#    python realtime_detection.py
#
# =============================================================================

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from collections import deque

# ─── CONFIGURATION ──────────────────────────────────────────
MODEL_PATH       = "drowsiness_model.keras"   # Path to your trained model
IMG_SIZE          = 128                         # Must match training size
EAR_THRESHOLD     = 0.22                       # Eye Aspect Ratio threshold
MAR_THRESHOLD     = 0.75                       # Mouth Aspect Ratio threshold
CNN_THRESHOLD     = 0.5                        # CNN drowsy probability threshold
DROWSY_FRAMES     = 15                         # Consecutive frames to trigger alert
SCORE_WEIGHTS     = {"eye": 0.5, "yawn": 0.3, "cnn": 0.2}
ALARM_COOLDOWN    = 5                          # Seconds between alarms


# ─── LOAD MODEL ─────────────────────────────────────────────
print("🔄 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
    USE_CNN = True
except Exception as e:
    print(f"⚠️  Model not found ({e}). Running with EAR/MAR only.")
    USE_CNN = False


# ─── MEDIAPIPE SETUP ────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for eyes and mouth (MediaPipe 468-point mesh)
# Left eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Mouth (outer lips)
UPPER_LIP  = [13]
LOWER_LIP  = [14]
LEFT_MOUTH  = [78]
RIGHT_MOUTH = [308]


# ─── HELPER FUNCTIONS ──────────────────────────────────────

def calculate_ear(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio (EAR) for one eye."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    # Vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    h1 = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if h1 == 0:
        return 0.3
    ear = (v1 + v2) / (2.0 * h1)
    return ear


def calculate_mar(landmarks, w, h):
    """Calculate Mouth Aspect Ratio (MAR)."""
    upper = np.array([int(landmarks[13].x * w), int(landmarks[13].y * h)])
    lower = np.array([int(landmarks[14].x * w), int(landmarks[14].y * h)])
    left  = np.array([int(landmarks[78].x * w), int(landmarks[78].y * h)])
    right = np.array([int(landmarks[308].x * w), int(landmarks[308].y * h)])

    mouth_open  = np.linalg.norm(upper - lower)
    mouth_width = np.linalg.norm(left - right)

    if mouth_width == 0:
        return 0.0
    return mouth_open / mouth_width


def predict_drowsiness(frame, model, img_size):
    """Run the CNN model on a face crop."""
    resized = cv2.resize(frame, (img_size, img_size))
    normalized = resized.astype("float32") / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)
    prediction = model.predict(input_tensor, verbose=0)[0][0]
    return prediction


def draw_fancy_box(frame, x1, y1, x2, y2, color, label, thickness=2):
    """Draw a styled bounding box with label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def draw_dashboard(frame, ear, mar, score, status, fps):
    """Draw a HUD-style dashboard overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent panel
    cv2.rectangle(overlay, (10, 10), (280, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "DRIVER MONITOR", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Metrics
    metrics = [
        (f"EAR:   {ear:.3f}", (0, 255, 0) if ear > EAR_THRESHOLD else (0, 0, 255)),
        (f"MAR:   {mar:.3f}", (0, 255, 0) if mar < MAR_THRESHOLD else (0, 165, 255)),
        (f"Score: {score:.2f}", (0, 255, 0) if score < 0.5 else (0, 0, 255)),
        (f"FPS:   {fps:.0f}", (200, 200, 200)),
    ]

    y = 65
    for text, color in metrics:
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 28

    # Status bar at top
    status_color = (0, 255, 0) if status == "ALERT" else \
                   (0, 165, 255) if status == "SLIGHTLY DROWSY" else (0, 0, 255)

    cv2.rectangle(frame, (0, h - 50), (w, h), status_color, -1)
    cv2.putText(frame, f"STATUS: {status}", (w // 2 - 120, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # EAR bar graph
    bar_x, bar_y, bar_w, bar_h = 20, 175, 200, 12
    fill = int(min(ear / 0.4, 1.0) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                  (0, 255, 0) if ear > EAR_THRESHOLD else (0, 0, 255), -1)


# ─── MAIN DETECTION LOOP ───────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State tracking
    drowsy_counter = 0
    yawn_counter   = 0
    blink_counter  = 0
    frame_count    = 0
    last_alarm     = 0
    score_history  = deque(maxlen=30)  # Smoothing window
    fps_history    = deque(maxlen=10)

    print("\n" + "=" * 50)
    print("  🚗 DRIVER DROWSINESS DETECTION — ACTIVE")
    print("  Press 'Q' to quit")
    print("=" * 50 + "\n")

    try:
        # Try to load pygame for audio alarm
        import pygame
        pygame.mixer.init()
        HAS_AUDIO = True
        print("🔊 Audio alarm enabled")
    except ImportError:
        HAS_AUDIO = False
        print("🔇 Install pygame for audio alerts: pip install pygame")

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = face_mesh.process(rgb_frame)

        ear = 0.3    # Default (eyes open)
        mar = 0.0    # Default (mouth closed)
        cnn_prob = 0.0
        status = "ALERT"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # ── Calculate EAR ──
            left_ear  = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            # ── Calculate MAR ──
            mar = calculate_mar(landmarks, w, h)

            # ── CNN Prediction (if model loaded) ──
            if USE_CNN:
                # Get face bounding box from landmarks
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                x1 = max(0, int(min(xs) * w) - 20)
                y1 = max(0, int(min(ys) * h) - 20)
                x2 = min(w, int(max(xs) * w) + 20)
                y2 = min(h, int(max(ys) * h) + 20)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    cnn_prob = predict_drowsiness(face_crop, model, IMG_SIZE)

            # ── Eye Closure Detection ──
            eye_score = 1.0 if ear < EAR_THRESHOLD else 0.0
            if ear < EAR_THRESHOLD:
                drowsy_counter += 1
            else:
                drowsy_counter = max(0, drowsy_counter - 1)

            # ── Yawn Detection ──
            yawn_score = 1.0 if mar > MAR_THRESHOLD else 0.0
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = max(0, yawn_counter - 1)

            # ── Composite Fatigue Score ──
            fatigue_score = (
                SCORE_WEIGHTS["eye"]  * eye_score +
                SCORE_WEIGHTS["yawn"] * yawn_score +
                SCORE_WEIGHTS["cnn"]  * (cnn_prob if USE_CNN else 0)
            )
            score_history.append(fatigue_score)
            avg_score = np.mean(score_history)

            # ── Determine Status ──
            if drowsy_counter > DROWSY_FRAMES or avg_score > 0.6:
                status = "DROWSY — WAKE UP!"
            elif drowsy_counter > DROWSY_FRAMES // 2 or avg_score > 0.3:
                status = "SLIGHTLY DROWSY"
            else:
                status = "ALERT"

            # ── Trigger Alarm ──
            if status == "DROWSY — WAKE UP!":
                current_time = time.time()
                if current_time - last_alarm > ALARM_COOLDOWN:
                    last_alarm = current_time
                    print("🚨 ALARM: Driver appears drowsy!")
                    if HAS_AUDIO:
                        # Generate a beep sound
                        try:
                            frequency = 2500
                            duration_ms = 1000
                            sample_rate = 44100
                            n_samples = int(sample_rate * duration_ms / 1000)
                            t = np.linspace(0, duration_ms / 1000, n_samples, False)
                            wave = np.sin(2 * np.pi * frequency * t) * 0.5
                            wave = (wave * 32767).astype(np.int16)
                            stereo_wave = np.column_stack((wave, wave))
                            sound = pygame.sndarray.make_sound(stereo_wave)
                            sound.play()
                        except Exception:
                            pass

            # ── Draw Eye Landmarks ──
            for eye_indices in [LEFT_EYE, RIGHT_EYE]:
                pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
                for pt in pts:
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            # ── Draw Face Box ──
            if USE_CNN:
                box_color = (0, 255, 0) if status == "ALERT" else \
                            (0, 165, 255) if status == "SLIGHTLY DROWSY" else (0, 0, 255)
                draw_fancy_box(frame, x1, y1, x2, y2, box_color,
                               f"CNN: {cnn_prob:.1%}")

        else:
            status = "NO FACE DETECTED"

        # ── FPS Calculation ──
        fps = 1.0 / (time.time() - start_time + 1e-6)
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)

        # ── Draw Dashboard ──
        draw_dashboard(frame, ear, mar,
                       np.mean(score_history) if score_history else 0,
                       status, avg_fps)

        # Show frame
        cv2.imshow("Driver Drowsiness Detection", frame)

        # Quit on 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Detection stopped.")


if __name__ == "__main__":
    main()
