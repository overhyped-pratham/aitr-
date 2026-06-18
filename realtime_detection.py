# =============================================================================
#  REAL-TIME DRIVER DROWSINESS DETECTION + AUTOMATIC SOS
#  Run this on your LOCAL machine after training on Colab.
#  Uses: OpenCV + MediaPipe + Your Trained Model + Twilio + Gemini
# =============================================================================
#
#  Setup (run once):
#    pip install opencv-python mediapipe numpy tensorflow pygame requests python-dotenv
#
#  Usage:
#    python realtime_detection.py
#
# =============================================================================

import sys
import cv2
import numpy as np
import mediapipe as mp
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
import time
import os
import threading
import requests
from collections import deque
from datetime import datetime

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load environment variables from .env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ─── CONFIGURATION ──────────────────────────────────────────
MODEL_PATH       = "drowsiness_model.keras"   # Path to your trained model
IMG_SIZE          = 128                         # Must match training size
EAR_THRESHOLD     = 0.22                       # Eye Aspect Ratio threshold
MAR_THRESHOLD     = 0.75                       # Mouth Aspect Ratio threshold
CNN_THRESHOLD     = 0.5                        # CNN drowsy probability threshold
DROWSY_FRAMES     = 15                         # Consecutive frames to trigger alert
SCORE_WEIGHTS     = {"eye": 0.5, "yawn": 0.3, "cnn": 0.2}
ALARM_COOLDOWN    = 5                          # Seconds between alarms

# SOS Configuration
WARNING_THRESHOLD_SEC  = 10   # Seconds of drowsiness before warning phase
CRITICAL_THRESHOLD_SEC = 20   # Seconds of drowsiness before SOS dispatch
EMERGENCY_CONTACT      = os.getenv("TWILIO_TO_NUMBER", "+919691424963")

# Twilio Credentials
TWILIO_ACCOUNT_SID          = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN            = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_MESSAGING_SERVICE_SID = os.getenv("TWILIO_MESSAGING_SERVICE_SID", "")

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ─── LOAD MODEL ─────────────────────────────────────────────
if HAS_TF:
    print("🔄 Loading model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
        USE_CNN = True
    except Exception as e:
        print(f"⚠️  Model not found ({e}). Running with EAR/MAR only.")
        USE_CNN = False
else:
    print("⚠️  TensorFlow not installed. Running with EAR/MAR only.")
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


# ─── SOS HELPER FUNCTIONS ─────────────────────────────────

def get_gps_coordinates():
    """Fetch approximate GPS coordinates via IP geolocation API."""
    try:
        resp = requests.get("http://ip-api.com/json/?fields=lat,lon,city,regionName,country", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "lat": data.get("lat", 0.0),
                "lon": data.get("lon", 0.0),
                "city": data.get("city", "Unknown"),
                "region": data.get("regionName", ""),
                "country": data.get("country", "")
            }
    except Exception as e:
        print(f"⚠️  GPS lookup failed: {e}")
    # Fallback: New Delhi coordinates
    return {"lat": 28.6139, "lon": 77.2090, "city": "New Delhi", "region": "Delhi", "country": "India"}


def generate_gemini_summary(ear, mar, score, drowsy_duration, gps):
    """Call the Gemini REST API directly to generate an emergency dispatch summary."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        maps_link = f"https://maps.google.com/?q={gps['lat']},{gps['lon']}"

        prompt = (
            "You are an emergency dispatch AI assistant. Generate a CONCISE emergency summary "
            "(max 100 words) for a drowsy driver alert.\n\n"
            f"Telemetry Data:\n"
            f"- Timestamp: {timestamp}\n"
            f"- Eye Aspect Ratio (EAR): {ear:.3f} (threshold: {EAR_THRESHOLD})\n"
            f"- Mouth Aspect Ratio (MAR): {mar:.3f} (threshold: {MAR_THRESHOLD})\n"
            f"- Fatigue Score: {score:.2f}/1.00\n"
            f"- Continuous Drowsiness Duration: {drowsy_duration:.0f} seconds\n"
            f"- Location: {gps['city']}, {gps['region']}, {gps['country']}\n"
            f"- GPS: {gps['lat']:.4f}, {gps['lon']:.4f}\n"
            f"- Maps: {maps_link}\n\n"
            "Write a brief emergency dispatch summary including:\n"
            "1. Severity assessment\n"
            "2. Driver state description\n"
            "3. Recommended emergency action\n"
            "Keep it concise and professional."
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            print(f"⚠️  Gemini API error [{resp.status_code}]: {resp.text[:200]}")
            raise Exception(f"API returned {resp.status_code}")
    except Exception as e:
        print(f"⚠️  Gemini summary failed: {e}")
        return f"AUTOMATED ALERT: Driver drowsiness detected for {drowsy_duration:.0f}s. EAR={ear:.3f}, Score={score:.2f}. Location: {gps['city']}. Immediate response required."


def send_twilio_sms(body):
    """Send an SMS via the Twilio REST API (no SDK needed)."""
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        data = {
            "To": EMERGENCY_CONTACT,
            "MessagingServiceSid": TWILIO_MESSAGING_SERVICE_SID,
            "Body": body
        }
        resp = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
        if resp.status_code in (200, 201):
            print(f"✅ SOS SMS sent to {EMERGENCY_CONTACT}")
            return True
        else:
            print(f"❌ SMS failed [{resp.status_code}]: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ SMS send error: {e}")
        return False


def play_beep(frequency=2500, duration_ms=500):
    """Generate and play a beep sound using pygame or winsound on Windows."""
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        sample_rate = 44100
        n_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n_samples, False)
        wave = np.sin(2 * np.pi * frequency * t) * 0.5
        wave = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.play()
        return
    except Exception:
        pass

    # Fallback to winsound on Windows
    if sys.platform == "win32":
        try:
            import winsound
            threading.Thread(target=winsound.Beep, args=(frequency, duration_ms), daemon=True).start()
        except Exception:
            pass


def draw_multiline_text(frame, text, origin, font, font_scale, color, thickness, max_width):
    """Draw word-wrapped text on an OpenCV frame. Returns the total height used."""
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (tw, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if tw > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    x, y = origin
    line_height = int(cv2.getTextSize("Tg", font, font_scale, thickness)[0][1] * 1.8)
    for line in lines:
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness)
        y += line_height
    return y - origin[1]


# ─── EXISTING HELPER FUNCTIONS ─────────────────────────────

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


def draw_dashboard(frame, ear, mar, score, status, fps, sos_status, gemini_summary):
    """Draw a HUD-style dashboard overlay with SOS info."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Left Panel: Metrics ──
    cv2.rectangle(overlay, (10, 10), (280, 240), (0, 0, 0), -1)
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

    # SOS Status line
    sos_color = (0, 255, 0)  # Green by default
    if "WARNING" in sos_status:
        sos_color = (0, 165, 255)  # Orange
    elif sos_status in ("SENDING...", "SENT", "CRITICAL"):
        sos_color = (0, 0, 255)  # Red
    elif sos_status == "FAILED":
        sos_color = (0, 0, 200)

    cv2.putText(frame, f"SOS: {sos_status}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, sos_color, 1)
    y += 28

    # EAR bar graph
    bar_x, bar_y, bar_w, bar_h = 20, y, 200, 12
    fill = int(min(ear / 0.4, 1.0) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                  (0, 255, 0) if ear > EAR_THRESHOLD else (0, 0, 255), -1)

    # ── Right Panel: Safety Hub & SOS Info ──
    panel_x = w - 310
    panel_y = 10
    panel_w = 300
    panel_h = 240

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "SAFETY HUB & SOS INFO", (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if gemini_summary:
        draw_multiline_text(
            frame, gemini_summary,
            (panel_x + 10, panel_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1,
            max_width=panel_w - 20
        )
    else:
        cv2.putText(frame, "No active alerts.", (panel_x + 10, panel_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(frame, "System monitoring...", (panel_x + 10, panel_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # ── Status bar at bottom ──
    if "CRITICAL" in status:
        status_color = (0, 0, 200)
    elif status == "DROWSY — WAKE UP!":
        status_color = (0, 0, 255)
    elif status == "SLIGHTLY DROWSY":
        status_color = (0, 165, 255)
    elif status == "ALERT":
        status_color = (0, 255, 0)
    else:
        status_color = (128, 128, 128)

    cv2.rectangle(frame, (0, h - 50), (w, h), status_color, -1)
    cv2.putText(frame, f"STATUS: {status}", (w // 2 - 150, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)


def draw_warning_popup(frame, countdown_sec):
    """Draw a centered emergency warning popup card on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent red overlay
    overlay = frame.copy()
    card_w, card_h = 500, 200
    cx, cy = w // 2, h // 2
    x1, y1 = cx - card_w // 2, cy - card_h // 2
    x2, y2 = cx + card_w // 2, cy + card_h // 2

    # Red card background
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
    # Border
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Warning text
    lines = [
        ("!!! EMERGENCY WARNING !!!", 0.7, (255, 255, 255), 2),
        ("CRITICAL DROWSINESS DETECTED", 0.55, (255, 200, 200), 1),
        (f"SOS DISPATCH IN: {countdown_sec:.0f} SECONDS", 0.65, (0, 255, 255), 2),
        ("Press [SPACEBAR] to Cancel Alert", 0.45, (200, 200, 200), 1),
    ]

    text_y = y1 + 45
    for text, scale, color, thick in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        text_x = cx - tw // 2
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
        text_y += 45


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

    # ── SOS State Variables ──
    drowsy_start_time = None   # When continuous drowsiness began
    sos_sent          = False  # One SMS per incident
    sos_status        = "IDLE"
    gemini_summary    = ""
    last_warning_beep = 0      # Timestamp of last warning beep
    sos_sending       = False  # True while background SOS thread is running

    print("\n" + "=" * 50)
    print("  🚗 DRIVER DROWSINESS DETECTION + SOS — ACTIVE")
    print("  Press 'Q' to quit | 'SPACEBAR' to dismiss alert")
    print("=" * 50 + "\n")

    # Check credentials
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print("⚠️  Twilio credentials not set in .env — SMS dispatch disabled")
    if not GEMINI_API_KEY:
        print("⚠️  Gemini API key not set in .env — AI summary disabled")

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

        # ════════════════════════════════════════════════════
        #  SOS DROWSINESS TIMER & DISPATCH LOGIC
        # ════════════════════════════════════════════════════

        is_drowsy = (status == "DROWSY — WAKE UP!")
        current_time = time.time()

        if is_drowsy:
            # Start or continue tracking continuous drowsiness
            if drowsy_start_time is None:
                drowsy_start_time = current_time

            drowsy_duration = current_time - drowsy_start_time

            # ── Phase 1: Regular drowsiness (0 - 10s) ──
            if drowsy_duration < WARNING_THRESHOLD_SEC:
                # Standard alarm beep every ALARM_COOLDOWN seconds
                if current_time - last_alarm > ALARM_COOLDOWN:
                    last_alarm = current_time
                    print("🚨 ALARM: Driver appears drowsy!")
                    if HAS_AUDIO:
                        play_beep(2500, 1000)

            # ── Phase 2: Warning phase (10s - 20s) ──
            elif drowsy_duration < CRITICAL_THRESHOLD_SEC:
                countdown = CRITICAL_THRESHOLD_SEC - drowsy_duration
                sos_status = f"WARNING: {countdown:.0f}s"
                status = "DROWSY — WAKE UP!"

                # Fast warning beeps every 0.8 seconds
                if current_time - last_warning_beep > 0.8:
                    last_warning_beep = current_time
                    if HAS_AUDIO:
                        play_beep(3500, 200)

                # Draw warning popup
                draw_warning_popup(frame, countdown)

            # ── Phase 3: Critical — SOS dispatch (>20s) ──
            elif not sos_sent and not sos_sending:
                status = "CRITICAL — SOS INITIATED!"
                sos_status = "SENDING..."
                sos_sending = True
                print("\n🆘 CRITICAL: Initiating emergency SOS dispatch...")

                # Capture current telemetry for the background thread
                _ear = ear
                _mar = mar
                _score = np.mean(score_history) if score_history else 0
                _duration = drowsy_duration

                def sos_dispatch():
                    nonlocal sos_sent, sos_status, gemini_summary, sos_sending
                    try:
                        # 1. Get GPS
                        print("📍 Fetching GPS coordinates...")
                        gps = get_gps_coordinates()
                        maps_link = f"https://maps.google.com/?q={gps['lat']},{gps['lon']}"
                        print(f"   Location: {gps['city']}, {gps['region']} ({gps['lat']:.4f}, {gps['lon']:.4f})")

                        # 2. Generate Gemini summary
                        print("🤖 Generating Gemini emergency summary...")
                        summary = generate_gemini_summary(_ear, _mar, _score, _duration, gps)
                        gemini_summary = summary
                        print(f"   Summary: {summary[:100]}...")

                        # 3. Compose and send SMS
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        sms_body = (
                            f"🚨 EMERGENCY ALERT — SafeDrive AI\n\n"
                            f"⏰ Time: {timestamp}\n"
                            f"📍 Location: {gps['city']}, {gps['region']}, {gps['country']}\n"
                            f"🗺️ Maps: {maps_link}\n"
                            f"📊 Fatigue Score: {_score:.2f} | EAR: {_ear:.3f}\n"
                            f"⏱️ Drowsy Duration: {_duration:.0f}s\n\n"
                            f"🤖 AI Assessment:\n{summary}\n\n"
                            f"⚠️ Driver unresponsive — immediate assistance may be required."
                        )

                        print("📤 Sending emergency SMS...")
                        success = send_twilio_sms(sms_body)

                        if success:
                            sos_sent = True
                            sos_status = "SENT"
                            print("✅ SOS dispatch complete!")
                        else:
                            sos_status = "FAILED"
                            print("❌ SOS dispatch failed!")
                    except Exception as e:
                        sos_status = "FAILED"
                        print(f"❌ SOS dispatch error: {e}")
                    finally:
                        sos_sending = False

                # Run SOS dispatch in background to avoid blocking camera loop
                threading.Thread(target=sos_dispatch, daemon=True).start()

            elif sos_sent:
                status = "CRITICAL — SOS SENT"
                sos_status = "SENT"

        else:
            # ── Driver is responsive — reset SOS state ──
            if drowsy_start_time is not None:
                drowsy_start_time = None
                if not sos_sent:
                    sos_status = "IDLE"
                # If SOS was sent, keep the status visible for a while
                # then reset after 30 seconds of being alert
                last_warning_beep = 0

            # Reset SOS sent flag if driver has been alert for 30+ seconds
            if sos_sent and sos_status == "SENT":
                if not hasattr(main, '_alert_resume_time') or main._alert_resume_time is None:
                    main._alert_resume_time = current_time
                elif current_time - main._alert_resume_time > 30:
                    sos_sent = False
                    sos_status = "IDLE"
                    gemini_summary = ""
                    main._alert_resume_time = None
                    print("✅ Driver alert — SOS state reset")
            else:
                if hasattr(main, '_alert_resume_time'):
                    main._alert_resume_time = None

        # ── FPS Calculation ──
        fps = 1.0 / (time.time() - start_time + 1e-6)
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)

        # ── Draw Dashboard ──
        draw_dashboard(frame, ear, mar,
                       np.mean(score_history) if score_history else 0,
                       status, avg_fps, sos_status, gemini_summary)

        # Show frame
        cv2.imshow("Driver Drowsiness Detection + SOS", frame)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF

        # Quit on 'Q'
        if key == ord('q'):
            break

        # Spacebar to dismiss alert
        if key == 32:  # Spacebar
            if drowsy_start_time is not None or sos_sent:
                print("⏸️  Alert dismissed by driver (SPACEBAR)")
                drowsy_start_time = None
                sos_sent = False
                sos_sending = False
                sos_status = "IDLE"
                gemini_summary = ""
                last_warning_beep = 0
                drowsy_counter = 0

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Detection stopped.")


if __name__ == "__main__":
    main()
