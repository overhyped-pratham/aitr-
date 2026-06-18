import os
import sys
import base64
import json
import time
import asyncio
import random
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# Ensure console supports unicode prints on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

try:
    from twilio.rest import Client as TwilioClient
    HAS_TWILIO = True
except ImportError:
    HAS_TWILIO = False

# Load environment variables
load_dotenv(find_dotenv())

app = FastAPI(title="DriveSafe AI Backend", description="Real-time fatigue detection and safety API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For hackathon ease, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIGURATION ──────────────────────────────────────────
IMG_SIZE = 128
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.70  # Lowered slightly for quicker yawn detection
CNN_THRESHOLD = 0.5
SCORE_WEIGHTS = {"eye": 0.5, "yawn": 0.3, "cnn": 0.2}

# ─── PACKAGE DETECTS ────────────────────────────────────────
HAS_OPENCV = False
HAS_NUMPY = False
HAS_MEDIAPIPE = False
HAS_TENSORFLOW = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    pass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    pass

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    pass

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    pass

HAS_REAL_TRACKING = HAS_OPENCV and HAS_NUMPY and HAS_MEDIAPIPE
print(f"[INFO] Package status: OpenCV={HAS_OPENCV}, NumPy={HAS_NUMPY}, MediaPipe={HAS_MEDIAPIPE}, TensorFlow={HAS_TENSORFLOW}")
print(f"[INFO] Real camera tracking: {'ENABLED' if HAS_REAL_TRACKING else 'DISABLED (FALLBACK TO MOCK)'}")
print(f"[INFO] CNN inference model: {'ENABLED' if HAS_TENSORFLOW else 'DISABLED (TENSORFLOW MISSING)'}")

# Global ML Objects
model = None
USE_CNN = False
face_mesh = None

# Eye & Mouth landmark indices (MediaPipe 468-point mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

if HAS_REAL_TRACKING:
    # ─── MEDIAPIPE SETUP ────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ─── MATH HELPERS ───────────────────────────────────────────
    def calculate_ear(landmarks, eye_indices, w, h):
        """Calculate Eye Aspect Ratio (EAR)."""
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h1 = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        if h1 == 0:
            return 0.3
        return (v1 + v2) / (2.0 * h1)

    def calculate_mar(landmarks, w, h):
        """Calculate Mouth Aspect Ratio (MAR)."""
        upper = np.array([int(landmarks[13].x * w), int(landmarks[13].y * h)])
        lower = np.array([int(landmarks[14].x * w), int(landmarks[14].y * h)])
        left  = np.array([int(landmarks[78].x * w), int(landmarks[78].y * h)])
        right = np.array([int(landmarks[308].x * w), int(landmarks[308].y * h)])

        mouth_open = np.linalg.norm(upper - lower)
        mouth_width = np.linalg.norm(left - right)

        if mouth_width == 0:
            return 0.0
        return mouth_open / mouth_width

    def calculate_pitch(landmarks, w, h):
        """Calculate vertical head droop (pitch)."""
        nose = landmarks[1].y * h
        forehead = landmarks[10].y * h
        chin = landmarks[152].y * h
        
        face_height = chin - forehead
        if face_height <= 0:
            return 0.5
        return (nose - forehead) / face_height

    def calculate_yaw(landmarks, w, h):
        """Calculate horizontal distraction / looking away (yaw)."""
        nose = landmarks[1].x * w
        left_cheek = landmarks[234].x * w
        right_cheek = landmarks[454].x * w
        
        face_width = right_cheek - left_cheek
        if face_width <= 0:
            return 0.5
        return (nose - left_cheek) / face_width

    def decode_base64_image(base64_str: str):
        """Decode base64 string to OpenCV BGR Image."""
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

if HAS_TENSORFLOW:
    # Locate model defensively
    MODEL_PATH = "drowsiness_model.keras"
    possible_paths = [
        "drowsiness_model.keras",
        "../drowsiness_model.keras",
        "model/drowsiness_model_final.keras",
        "../model/drowsiness_model_final.keras",
        "backend/../drowsiness_model.keras",
        "backend/model/drowsiness_model_final.keras"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"[INFO] Loading trained model from: {path}")
            try:
                model = tf.keras.models.load_model(path, compile=False)
                print(f"[OK] Model loaded successfully from {path}!")
                USE_CNN = True
                break
            except Exception as e:
                print(f"[WARNING] Failed to load model from {path}: {e}")

    if not USE_CNN:
        print("[WARNING] No trained Keras model loaded. Drowsiness detection will rely on EAR and MAR calculations only.")

# ─── API MODELS ─────────────────────────────────────────────
class RiskRequest(BaseModel):
    fatigueScore: float
    tripDurationMin: int
    timeOfDay: str  # e.g., "Night", "Afternoon", "Morning"
    customApiKey: Optional[str] = None

class NearbyRequest(BaseModel):
    latitude: float
    longitude: float
    queryType: str  # "hospital", "fuel", "rest_stop"
    customApiKey: Optional[str] = None

class SMSRequest(BaseModel):
    to_phone: Optional[str] = None
    message: str

class EmergencySummaryRequest(BaseModel):
    fatigueScore: float
    ear: float
    mar: float
    blinkCount: int
    yawnCount: int
    distractionDuration: float
    latitude: float
    longitude: float
    customApiKey: Optional[str] = None

# ─── GEMINI RISK ANALYSIS ───────────────────────────────────
@app.post("/api/risk-analysis")
async def analyze_risk(data: RiskRequest):
    # Determine API key to use
    api_key = data.customApiKey or os.getenv("GEMINI_API_KEY")
    
    prompt = f"""
    You are an AI Driver Safety Assistant for "DriveSafe AI". 
    Analyze the following driver telemetry and provide a detailed risk explanation and actionable safety recommendations:
    - Current Fatigue Score: {data.fatigueScore:.1f}%
    - Trip Duration: {data.tripDurationMin} minutes
    - Time of Day: {data.timeOfDay}
    
    Structure your response with:
    1. A brief "Risk Severity" rating (e.g., Low, Medium, High, Critical).
    2. A structured explanation of what this fatigue level means (e.g., reaction delay, microsleep risk).
    3. Exactly 3-4 bulleted, immediate recommendations (e.g., stop details, exercises, cognitive tasks).
    
    Keep the response concise, clear, and easy to read on a driver's dashboard. Avoid markdown headers like # or ##, just use bold text and standard spacing.
    """
    
    if not api_key or api_key == "" or api_key.startswith("your_") or len(api_key) < 10:
        # Graceful fallback: Mock Gemini response
        time.sleep(1.0)  # Simulate API latency
        
        severity = "LOW"
        if data.fatigueScore > 75:
            severity = "CRITICAL"
        elif data.fatigueScore > 40:
            severity = "HIGH"
        elif data.fatigueScore > 20:
            severity = "MEDIUM"
            
        explanation = ""
        recommendations = []
        
        if severity == "CRITICAL":
            explanation = f"CRITICAL DANGER: A fatigue score of {data.fatigueScore:.1f}% indicates active signs of sleepiness, including prolonged eye closures and frequent yawning. You are at high risk of microsleep, which can lead to lane departure in less than 2 seconds. Your reaction speed is impaired by approximately 50%."
            recommendations = [
                "PULL OVER IMMEDIATELY: Find the nearest rest stop or highway exit. Do not try to push through.",
                "Drink coffee or a caffeinated beverage, but remember it takes 20 minutes to take effect and is only a temporary fix.",
                "Take a 20-minute power nap. Studies show power naps restore alertness effectively.",
                "Contact an emergency contact or co-driver to take over the wheel."
            ]
        elif severity == "HIGH":
            explanation = f"HIGH RISK: At {data.fatigueScore:.1f}% fatigue, your blinking is slowing down and you are showing signs of heavy yawning. Distraction and momentary lapses in concentration are highly probable. Your reaction time is delayed."
            recommendations = [
                "Plan a stop within the next 10-15 minutes.",
                "Roll down the windows to let cold, fresh air circulate inside the cabin.",
                "Play fast-paced, high-energy music or talk out loud to stimulate brain activity.",
                "Wash your face with cold water at the next opportunity."
            ]
        elif severity == "MEDIUM":
            explanation = f"MODERATE FATIGUE: Fatigue is at {data.fatigueScore:.1f}%. Mild tiredness detected, particularly common during {data.timeOfDay} driving. Reaction times may be slightly affected."
            recommendations = [
                "Drink some cold water to stay hydrated and refresh your system.",
                "Adjust your posture and do simple shoulder shrugs or neck rolls.",
                "Lower the vehicle's AC temperature by a few degrees to keep yourself cool.",
                "Engage in simple conversation if you have passengers."
            ]
        else:
            explanation = f"SAFE STATUS: Fatigue score is {data.fatigueScore:.1f}%. You are exhibiting alert driving behaviors. Keep monitoring."
            recommendations = [
                "Keep driving safely and maintain your current pace.",
                "Ensure correct seat adjustment and mirror visibility.",
                "Remember to plan a rest break every 2 hours or 100 miles."
            ]
            
        return {
            "source": "Mock Engine (No Gemini API Key provided)",
            "severity": severity,
            "explanation": explanation,
            "recommendations": recommendations
        }
        
    try:
        from google import genai
        # Initialize client with specified API Key
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        # Parse the response text
        text = response.text
        
        # Basic parsing to extract severity
        severity = "LOW"
        if "critical" in text.lower():
            severity = "CRITICAL"
        elif "high" in text.lower():
            severity = "HIGH"
        elif "medium" in text.lower() or "moderate" in text.lower():
            severity = "MEDIUM"
            
        # Clean response and format
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        explanation_lines = []
        recommendations = []
        
        for line in lines:
            if line.startswith("-") or line.startswith("*") or any(line.startswith(f"{i}.") for i in range(1, 10)) or "recommend" in line.lower() and ":" in line:
                cleaned = line.lstrip("-* \t").strip()
                if cleaned and not cleaned.lower().startswith("recommend"):
                    recommendations.append(cleaned)
            else:
                if "severity" not in line.lower():
                    explanation_lines.append(line)
                    
        explanation = " ".join(explanation_lines[:3]) if explanation_lines else "AI analysis completed."
        
        if not recommendations:
            recommendations = [
                "Pull over at the nearest safe rest area.",
                "Take a 15-20 minute power nap.",
                "Engage active cooling (AC or open window) and hydrate."
            ]
            
        return {
            "source": "Gemini 2.5 Flash",
            "severity": severity,
            "explanation": explanation,
            "recommendations": recommendations[:4]
        }
    except Exception as e:
        print(f"[ERROR] Gemini API Error: {e}")
        return {
            "source": "Error Fallback Engine",
            "severity": "HIGH",
            "explanation": f"Failed to call Gemini API: {str(e)}. Please check your API key configuration.",
            "recommendations": [
                "PULL OVER IMMEDIATELY: Take a break from driving.",
                "Ensure your .env file contains a valid GEMINI_API_KEY.",
                "Drink water and rest your eyes."
            ]
        }

# ─── EMERGENCY ALERTS & SMS ENDPOINTS ────────────────────────
@app.post("/api/send-sms")
async def send_sms(data: SMSRequest):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_phone = os.getenv("TWILIO_PHONE_NUMBER")
    messaging_service_sid = os.getenv("TWILIO_MESSAGING_SERVICE_SID")
    default_to = os.getenv("TWILIO_TO_NUMBER")
    
    to_number = data.to_phone or default_to
    
    has_sender = bool(from_phone or messaging_service_sid)
    if not account_sid or not auth_token or not has_sender or not to_number or not HAS_TWILIO:
        # Fallback to mock log
        print(f"[SMS LOG (MOCK)] To: {to_number or 'Unconfigured'}\nMessage:\n{data.message}")
        return {
            "status": "mocked",
            "message": "Twilio not configured. Message logged to console and displayed in UI.",
            "to": to_number or "Unconfigured",
            "sms_body": data.message
        }
    
    try:
        client = TwilioClient(account_sid, auth_token)
        send_params = {
            "body": data.message,
            "to": to_number
        }
        if messaging_service_sid:
            send_params["messaging_service_sid"] = messaging_service_sid
        else:
            send_params["from_"] = from_phone
            
        message = client.messages.create(**send_params)
        return {
            "status": "sent",
            "sid": message.sid,
            "message": "SMS alert sent successfully via Twilio!",
            "to": to_number
        }
    except Exception as e:
        print(f"[ERROR] Twilio send failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "message": f"Twilio API Error: {str(e)}",
            "to": to_number,
            "sms_body": data.message
        }

@app.post("/api/emergency-summary")
async def emergency_summary(data: EmergencySummaryRequest):
    api_key = data.customApiKey or os.getenv("GEMINI_API_KEY")
    
    prompt = f"""
    You are an Emergency Response AI Dispatcher for SafeDrive AI. 
    Analyze this driver telemetry and generate a structured rescue report for emergency dispatch operators:
    
    DRIVER TELEMETRY STATUS:
    - Current Fatigue Score: {data.fatigueScore:.1f}% (CRITICAL)
    - Eye Aspect Ratio (EAR): {data.ear:.3f}
    - Blink Rate: {data.blinkCount} blinks in session
    - Yawn Rate: {data.yawnCount} yawns in session
    - Distraction Duration: {data.distractionDuration} seconds looking away
    - Last Coordinates: Latitude {data.latitude:.6f}, Longitude {data.longitude:.6f}
    
    Generate a response in the following format:
    1. Incident Hazard: (2 sentences summarizing the driver's current condition and the immediate risk of a crash)
    2. Telemetry Status: (1-2 sentences explaining the critical parameters like eye closure or yawning)
    3. Recommended Rescue Action: (Provide coordinates and suggest sending emergency assistance or directing the driver to the nearest rest stop)
    
    Keep the response extremely brief, clear, and highly urgent. Do not use markdown headers (# or ##), use bold text and standard lists.
    """
    
    if not api_key or api_key == "" or api_key.startswith("your_") or len(api_key) < 10:
        # Graceful fallback: Mock Gemini response
        time.sleep(1.0)
        return {
            "source": "Mock Emergency Engine",
            "incident_hazard": "CRITICAL HAZARD: Driver has exhibited sustained eye closure and unresponsive behavior for over 10 seconds. Immediate crash danger due to vehicle drift.",
            "telemetry_status": f"Telemetry shows fatigue index at {data.fatigueScore:.1f}%, EAR at {data.ear:.3f}, with {data.yawnCount} yawning events and {data.distractionDuration}s of horizontal distraction.",
            "recommended_action": f"DISPATCH EMERGENCY SERVICES to Latitude {data.latitude:.6f}, Longitude {data.longitude:.6f} (Google Maps link: https://maps.google.com/?q={data.latitude},{data.longitude}). Route driver to the nearest highway shoulder or rest stop."
        }
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        text = response.text
        
        # Parse into sections or just return the text
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        hazard = "Driver fatigue detected with prolonged eye closure and no response to alarms."
        telemetry = f"Fatigue score: {data.fatigueScore:.1f}%, EAR: {data.ear:.3f}, yawns: {data.yawnCount}."
        action = f"Immediate roadside intervention recommended. Coordinates: {data.latitude}, {data.longitude}."
        
        current_section = 0
        hazard_lines = []
        telemetry_lines = []
        action_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if "hazard" in line_lower or "incident" in line_lower:
                current_section = 1
                continue
            elif "telemetry" in line_lower or "status" in line_lower:
                current_section = 2
                continue
            elif "action" in line_lower or "recommend" in line_lower or "dispatch" in line_lower:
                current_section = 3
                continue
                
            if current_section == 1:
                hazard_lines.append(line)
            elif current_section == 2:
                telemetry_lines.append(line)
            elif current_section == 3:
                action_lines.append(line)
        
        if hazard_lines:
            hazard = " ".join(hazard_lines)
        if telemetry_lines:
            telemetry = " ".join(telemetry_lines)
        if action_lines:
            action = " ".join(action_lines)
        else:
            action = text[:400]
            
        return {
            "source": "Gemini 2.5 Flash",
            "incident_hazard": hazard,
            "telemetry_status": telemetry,
            "recommended_action": action
        }
    except Exception as e:
        print(f"[ERROR] Gemini Emergency Summary Error: {e}")
        return {
            "source": "Error Fallback Engine",
            "incident_hazard": "CRITICAL Driver Fatigue and potential sleep state detected.",
            "telemetry_status": f"FastAPI Server report failed: {str(e)}.",
            "recommended_action": f"Dispatch highway patrol to coordinate check at Lat {data.latitude}, Lng {data.longitude}."
        }

# ─── GOOGLE PLACES MOCK / QUERY ─────────────────────────────
@app.post("/api/send-sos")
async def send_sos(data: EmergencySummaryRequest):
    # Determine keys
    api_key = data.customApiKey or os.getenv("GEMINI_API_KEY")
    maps_key = os.getenv("NEXT_PUBLIC_GOOGLE_MAPS_API_KEY")
    
    # 1. Fetch nearest Hospital
    hospital = "City Emergency Hospital"
    hospital_address = "450 Medical Center Blvd"
    if maps_key and not maps_key.startswith("your_") and len(maps_key) > 10:
        try:
            import httpx
            # Query hospital
            url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": f"{data.latitude},{data.longitude}",
                "radius": "10000",
                "type": "hospital",
                "key": maps_key
            }
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params)
                res = resp.json().get("results", [])
                if res:
                    hospital = res[0].get("name", "City Emergency Hospital")
                    hospital_address = res[0].get("vicinity", "450 Medical Center Blvd")
        except Exception as e:
            print(f"[ERROR] Places API failed in SOS: {e}")
            
    # 2. Fetch nearest Police Station
    police_station = "Highway Police Station HQ"
    police_address = "220 Highway Patrol Blvd"
    if maps_key and not maps_key.startswith("your_") and len(maps_key) > 10:
        try:
            import httpx
            # Query police
            url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": f"{data.latitude},{data.longitude}",
                "radius": "10000",
                "type": "police",
                "key": maps_key
            }
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params)
                res = resp.json().get("results", [])
                if res:
                    police_station = res[0].get("name", "Highway Police Station HQ")
                    police_address = res[0].get("vicinity", "220 Highway Patrol Blvd")
        except Exception as e:
            print(f"[ERROR] Places API failed in SOS: {e}")

    # 3. Generate Gemini emergency report (Module 1 Role)
    prompt = f"""
    You are an Emergency Response AI Dispatcher for SafeDrive AI. 
    Analyze this driver telemetry and generate a structured rescue report for emergency dispatch operators:
    
    DRIVER STATUS:
    - Current Fatigue Score: {data.fatigueScore:.1f}% (CRITICAL)
    - Eye Aspect Ratio (EAR): {data.ear:.3f}
    - Blink Rate: {data.blinkCount}
    - Yawn Rate: {data.yawnCount}
    - Distraction Duration: {data.distractionDuration} seconds
    - Last Coordinates: Latitude {data.latitude:.6f}, Longitude {data.longitude:.6f}
    
    Generate a response in the following format:
    1. Incident Hazard: (2 sentences summarizing driver's condition and risk)
    2. Telemetry Status: (1 sentence explaining EAR and MAR)
    3. Recommended Rescue Action: (Provide coordinates and suggest calling rescue services)
    
    Keep the report extremely brief and clear. Do not use markdown headers (like # or ##).
    """
    
    report_summary = ""
    if not api_key or api_key.startswith("your_") or len(api_key) < 10:
        report_summary = "Critical fatigue detected. The driver has shown prolonged eye closure and multiple fatigue indicators. Immediate intervention is recommended. The nearest medical facility should be contacted if the driver remains unresponsive."
    else:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            report_summary = response.text.strip()
        except Exception as e:
            print(f"[ERROR] Gemini failed in SOS: {e}")
            report_summary = "Critical driver fatigue detected. Telemetry suggests severe drowsiness or microsleep. Immediate rescue dispatch recommended."

    # 4. Generate SOS message body exactly like standard specs
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    google_maps_link = f"https://maps.google.com/?q={data.latitude:.6f},{data.longitude:.6f}"
    
    sos_message = f"""EMERGENCY ALERT

Critical driver fatigue detected.

Location:
Latitude: {data.latitude:.6f}
Longitude: {data.longitude:.6f}

Google Maps Link:
{google_maps_link}

Nearest Hospital:
{hospital}

Nearest Police Station:
{police_station}

Time:
{timestamp}"""

    # 5. Broadcast to Twilio recipients
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_phone = os.getenv("TWILIO_PHONE_NUMBER")
    messaging_service_sid = os.getenv("TWILIO_MESSAGING_SERVICE_SID")
    
    to_numbers = [os.getenv("TWILIO_TO_NUMBER")]
    to_number_2 = os.getenv("TWILIO_TO_NUMBER_2")
    if to_number_2:
        to_numbers.append(to_number_2)
        
    sms_status = "mocked"
    sms_message = "Twilio not configured. Message logged to console."
    
    has_sender = bool(from_phone or messaging_service_sid)
    if account_sid and auth_token and has_sender and HAS_TWILIO:
        sms_status = "sent"
        sms_message = "SOS message successfully broadcasted to emergency contacts!"
        for to_num in to_numbers:
            if not to_num:
                continue
            try:
                client = TwilioClient(account_sid, auth_token)
                send_params = {
                    "body": sos_message,
                    "to": to_num
                }
                if messaging_service_sid:
                    send_params["messaging_service_sid"] = messaging_service_sid
                else:
                    send_params["from_"] = from_phone
                    
                client.messages.create(**send_params)
            except Exception as e:
                print(f"[ERROR] Twilio broadcast failed to {to_num}: {e}")
                sms_status = "partial_failure"
                sms_message = f"Failed to send to some contacts: {str(e)}"
    else:
        print(f"[SMS LOG (MOCK SOS BROADCAST)]\nMessage:\n{sos_message}")

    return {
        "status": "success",
        "sos_message": sos_message,
        "emergency_summary": report_summary,
        "sms_status": sms_status,
        "sms_feedback": sms_message,
        "hospital": {
            "name": hospital,
            "address": hospital_address,
            "maps_link": f"https://www.google.com/maps/search/?api=1&query={data.latitude+0.008},{data.longitude-0.005}" if (not maps_key) else f"https://www.google.com/maps/search/?api=1&query={hospital.replace(' ', '+')}"
        },
        "police_station": {
            "name": police_station,
            "address": police_address,
            "maps_link": f"https://www.google.com/maps/search/?api=1&query={data.latitude-0.009},{data.longitude-0.004}" if (not maps_key) else f"https://www.google.com/maps/search/?api=1&query={police_station.replace(' ', '+')}"
        }
    }

@app.post("/api/nearby-assistance")
async def get_nearby_assistance(data: NearbyRequest):
    # This endpoint can call Google Places API if a key is provided
    api_key = data.customApiKey or os.getenv("NEXT_PUBLIC_GOOGLE_MAPS_API_KEY")
    
    lat = data.latitude
    lng = data.longitude
    q_type = data.queryType
    
    is_live = api_key is not None and api_key != "" and not api_key.startswith("your_") and len(api_key) > 10
    
    if is_live:
        try:
            import httpx
            # Map queryType to Google Places Types
            place_type = "hospital"
            if q_type == "police":
                place_type = "police"
            elif q_type == "fuel":
                place_type = "gas_station"
            elif q_type == "rest_stop":
                place_type = "lodging"
                
            url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": f"{lat},{lng}",
                "radius": "10000",
                "type": place_type,
                "key": api_key
            }
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params)
                res_data = resp.json()
                results = res_data.get("results", [])
                
                places = []
                for p in results[:3]:
                    p_lat = p["geometry"]["location"]["lat"]
                    p_lng = p["geometry"]["location"]["lng"]
                    places.append({
                        "name": p.get("name"),
                        "lat": p_lat,
                        "lng": p_lng,
                        "distance": "Within 5 km",
                        "duration": "5 min",
                        "phone": "N/A",
                        "address": p.get("vicinity", "N/A"),
                        "open_24h": p.get("opening_hours", {}).get("open_now", True)
                    })
                if places:
                    return {
                        "status": "success",
                        "query": q_type,
                        "api_key_configured": True,
                        "places": places
                    }
        except Exception as e:
            print(f"[ERROR] Live Places API query failed, falling back to mock: {e}")

    # Generate mock data based on query type
    mock_places = []
    
    if q_type == "hospital":
        mock_places = [
            {
                "name": "City Emergency Hospital",
                "lat": lat + 0.008,
                "lng": lng - 0.005,
                "distance": "1.2 km",
                "duration": "3 min",
                "phone": "+1 (555) 019-9000",
                "address": "450 Medical Center Blvd",
                "open_24h": True
            },
            {
                "name": "St. Jude Trauma Clinic",
                "lat": lat - 0.012,
                "lng": lng + 0.015,
                "distance": "2.8 km",
                "duration": "6 min",
                "phone": "+1 (555) 019-9122",
                "address": "88 Health Circle Road",
                "open_24h": True
            }
        ]
    elif q_type == "police":
        mock_places = [
            {
                "name": "Highway Police Station HQ",
                "lat": lat + 0.005,
                "lng": lng + 0.010,
                "distance": "1.8 km",
                "duration": "4 min",
                "phone": "+1 (555) 011-9922",
                "address": "220 Highway Patrol Blvd",
                "open_24h": True
            },
            {
                "name": "City Police Station Division 3",
                "lat": lat - 0.009,
                "lng": lng - 0.004,
                "distance": "2.4 km",
                "duration": "5 min",
                "phone": "+1 (555) 011-9988",
                "address": "150 North Main Street",
                "open_24h": True
            }
        ]
    elif q_type == "fuel":
        mock_places = [
            {
                "name": "Shell Highway Station",
                "lat": lat + 0.004,
                "lng": lng + 0.006,
                "distance": "0.7 km",
                "duration": "2 min",
                "phone": "+1 (555) 012-3344",
                "address": "1200 Expressway Route 6",
                "open_24h": True
            },
            {
                "name": "Chevron Express & Food Mart",
                "lat": lat - 0.006,
                "lng": lng - 0.008,
                "distance": "1.1 km",
                "duration": "3 min",
                "phone": "+1 (555) 012-5566",
                "address": "1480 Westside Parkway",
                "open_24h": True
            }
        ]
    else:  # rest_stop
        mock_places = [
            {
                "name": "Highway Rest Area 24",
                "lat": lat + 0.018,
                "lng": lng + 0.022,
                "distance": "3.5 km",
                "duration": "7 min",
                "phone": "N/A",
                "address": "Mile Marker 142 Eastbound",
                "open_24h": True
            },
            {
                "name": "Pilot Travel Center & Lounge",
                "lat": lat - 0.015,
                "lng": lng - 0.010,
                "distance": "2.1 km",
                "duration": "5 min",
                "phone": "+1 (555) 015-7788",
                "address": "300 Highway Interchange 4",
                "open_24h": True
            }
        ]
        
    return {
        "status": "success",
        "query": q_type,
        "api_key_configured": is_live,
        "places": mock_places
    }

# ─── WEBSOCKET FOR REAL-TIME DETECTION ──────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] WebSocket Client Connected!")
    
    # Frame rate smoothing and counters
    drowsy_counter = 0
    score_history = []
    frame_count = 0
    
    # Blinks, Yawns, Head Tilt, Distraction session counters
    blink_count = 0
    yawn_count = 0
    eye_closed_prev = False
    yawn_active_prev = False
    
    distracted_counter = 0
    droop_counter = 0
    distraction_duration = 0.0
    
    try:
        while True:
            # Receive frame data (JSON: {"image": "data:image/jpeg;base64,...", "settings": {...}})
            data_str = await websocket.receive_text()
            payload = json.loads(data_str)
            
            frame_count += 1
            
            if not HAS_REAL_TRACKING:
                # ── MOCK TELEMETRY ENGINE ──
                # Generate mock data that fluctuates realistically over time
                
                # Normal blinks (1 frame closure) every 22 frames
                is_blink = (frame_count % 22) in [0]
                
                # Simulate a yawn every 160 frames (lasts 10 frames)
                is_yawn = (frame_count % 160) in range(60, 70)
                
                # Simulate a critical drowsy micro-sleep event every 320 frames (lasts 22 frames)
                is_microsleep = (frame_count % 320) in range(180, 202)
                
                # Simulate distraction looking away every 200 frames (lasts 15 frames)
                is_distracted = (frame_count % 200) in range(110, 125)
                
                # Simulate head droop every 240 frames (lasts 15 frames)
                is_droop = (frame_count % 240) in range(160, 175)
                
                if is_microsleep:
                    ear = 0.12 + random.uniform(-0.01, 0.01)
                    mar = 0.15 + random.uniform(-0.02, 0.02)
                    cnn_prob = 0.91 + random.uniform(-0.04, 0.04)
                    drowsy_counter += 1
                elif is_blink:
                    ear = 0.14 + random.uniform(-0.02, 0.02)
                    mar = 0.12 + random.uniform(-0.02, 0.02)
                    cnn_prob = 0.08 + random.uniform(-0.03, 0.03)
                    drowsy_counter = max(0, drowsy_counter - 1)
                elif is_yawn:
                    ear = 0.28 + random.uniform(-0.02, 0.02)
                    mar = 0.82 + random.uniform(-0.05, 0.05)
                    cnn_prob = 0.48 + random.uniform(-0.05, 0.05)
                    drowsy_counter = max(0, drowsy_counter - 1)
                else:
                    ear = 0.32 + random.uniform(-0.02, 0.02)
                    mar = 0.12 + random.uniform(-0.02, 0.02)
                    cnn_prob = 0.03 + random.uniform(-0.02, 0.02)
                    drowsy_counter = max(0, drowsy_counter - 1)
                
                # Head droop pitch
                if is_droop:
                    pitch = 0.68 + random.uniform(0.01, 0.04)
                    droop_counter += 1
                else:
                    pitch = 0.52 + random.uniform(-0.02, 0.02)
                    droop_counter = max(0, droop_counter - 1)
                    
                # Distraction yaw (horizontal turn)
                if is_distracted:
                    yaw = 0.67 + random.uniform(0.01, 0.03)
                    distracted_counter += 1
                    distraction_duration += 0.12
                else:
                    yaw = 0.50 + random.uniform(-0.01, 0.01)
                    distracted_counter = max(0, distracted_counter - 1)
                
                # Dynamic session counters for blinks & yawns
                eye_closed = ear < EAR_THRESHOLD
                if eye_closed and not eye_closed_prev:
                    blink_count += 1
                eye_closed_prev = eye_closed
                
                yawning = mar > MAR_THRESHOLD
                if yawning and not yawning_prev:
                    yawn_count += 1
                yawning_prev = yawning
                
                eye_score = 1.0 if ear < EAR_THRESHOLD else 0.0
                yawn_score = 1.0 if mar > MAR_THRESHOLD else 0.0
                
                fatigue_score = (
                    SCORE_WEIGHTS["eye"]  * eye_score +
                    SCORE_WEIGHTS["yawn"] * yawn_score +
                    SCORE_WEIGHTS["cnn"]  * cnn_prob
                )
                
                score_history.append(fatigue_score)
                if len(score_history) > 15:
                    score_history.pop(0)
                avg_score = float(sum(score_history) / len(score_history))
                
                # Determine status
                if drowsy_counter > 10 or avg_score > 0.65:
                    status = "DROWSY"
                elif droop_counter > 8:
                    status = "HEAD DROOPING"
                elif distracted_counter > 8:
                    status = "DISTRACTED"
                elif drowsy_counter > 5 or avg_score > 0.30:
                    status = "SLIGHTLY DROWSY"
                else:
                    status = "ALERT"
                
                # Bounding box
                bbox = [180, 120, 460, 360]
                
                # Draw mock landmarks (scales eye/mouth aperture dynamically)
                ear_offset = 0.002 if (is_microsleep or is_blink) else 0.015
                mouth_offset = 0.05 if is_yawn else 0.01
                
                landmarks_list = [
                    # Left Eye
                    {"index": 362, "x": 0.40, "y": 0.38},
                    {"index": 385, "x": 0.42, "y": 0.36 - ear_offset},
                    {"index": 387, "x": 0.44, "y": 0.36 - ear_offset},
                    {"index": 263, "x": 0.46, "y": 0.38},
                    {"index": 373, "x": 0.44, "y": 0.40 + ear_offset},
                    {"index": 380, "x": 0.42, "y": 0.40 + ear_offset},
                    
                    # Right Eye
                    {"index": 33,  "x": 0.54, "y": 0.38},
                    {"index": 160, "x": 0.56, "y": 0.36 - ear_offset},
                    {"index": 158, "x": 0.58, "y": 0.36 - ear_offset},
                    {"index": 133, "x": 0.60, "y": 0.38},
                    {"index": 153, "x": 0.58, "y": 0.40 + ear_offset},
                    {"index": 144, "x": 0.56, "y": 0.40 + ear_offset},
                    
                    # Mouth
                    {"index": 13,  "x": 0.50, "y": 0.58 - mouth_offset},
                    {"index": 14,  "x": 0.50, "y": 0.58 + mouth_offset},
                    {"index": 78,  "x": 0.45, "y": 0.58},
                    {"index": 308, "x": 0.55, "y": 0.58}
                ]
                
                await websocket.send_json({
                    "ear": float(ear),
                    "mar": float(mar),
                    "cnn_prob": float(cnn_prob),
                    "fatigue_score": float(avg_score),
                    "status": f"{status} (MOCK)",
                    "bbox": bbox,
                    "landmarks": landmarks_list,
                    "use_cnn": False,
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                    "blink_count": blink_count,
                    "yawn_count": yawn_count,
                    "distraction_duration": float(distraction_duration)
                })
                continue
                
            # ─── REAL DETECTION MODE ───
            base64_frame = payload.get("image")
            if not base64_frame:
                continue
                
            try:
                frame = decode_base64_image(base64_frame)
            except Exception as e:
                await websocket.send_json({"error": f"Image decode failed: {str(e)}"})
                continue
                
            if frame is None or frame.size == 0:
                await websocket.send_json({"error": "Decoded frame is empty"})
                continue
                
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = face_mesh.process(rgb_frame)
            
            ear = 0.3
            mar = 0.0
            cnn_prob = 0.0
            landmarks_list = []
            bbox = []
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Calculate EAR
                left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
                right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0
                
                # Calculate MAR
                mar = calculate_mar(landmarks, w, h)
                
                # Calculate Pitch (head droop) and Yaw (looking away)
                pitch = calculate_pitch(landmarks, w, h)
                yaw = calculate_yaw(landmarks, w, h)
                
                # Crop face and run CNN model inference
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                x1 = max(0, int(min(xs) * w) - 15)
                y1 = max(0, int(min(ys) * h) - 15)
                x2 = min(w, int(max(xs) * w) + 15)
                y2 = min(h, int(max(ys) * h) + 15)
                bbox = [x1, y1, x2, y2]
                
                if USE_CNN and model is not None:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        try:
                            resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                            normalized = resized.astype("float32") / 255.0
                            input_tensor = np.expand_dims(normalized, axis=0)
                            
                            pred = model.predict(input_tensor, verbose=0)
                            cnn_prob = float(pred[0][0])
                        except Exception as e:
                            print(f"[WARNING] CNN Inference Error: {e}")
                            cnn_prob = 0.0
                else:
                    # Smart simulation fallback for CNN when model is not loaded (TensorFlow offline)
                    # This simulates a drowsiness probability based on EAR and MAR so the gauge is not stuck at 0%
                    if ear < EAR_THRESHOLD:
                        # Eye closed -> High probability of drowsiness
                        cnn_prob = 0.82 + random.uniform(0.01, 0.08)
                    elif mar > MAR_THRESHOLD:
                        # Yawning -> Elevated probability of drowsiness
                        cnn_prob = 0.55 + random.uniform(-0.05, 0.05)
                    else:
                        # Alert -> Very low probability
                        cnn_prob = 0.04 + random.uniform(-0.02, 0.02)
                    cnn_prob = max(0.0, min(1.0, cnn_prob))
                            
                # Eye closure counter
                eye_score = 1.0 if ear < EAR_THRESHOLD else 0.0
                if ear < EAR_THRESHOLD:
                    drowsy_counter += 1
                else:
                    drowsy_counter = max(0, drowsy_counter - 1)
                    
                # Yawn score
                yawn_score = 1.0 if mar > MAR_THRESHOLD else 0.0
                
                # Dynamic session counters for blinks & yawns
                eye_closed = ear < EAR_THRESHOLD
                if eye_closed and not eye_closed_prev:
                    blink_count += 1
                eye_closed_prev = eye_closed
                
                yawning = mar > MAR_THRESHOLD
                if yawning and not yawn_active_prev:
                    yawn_count += 1
                yawn_active_prev = yawning
                
                # Head droop chin tilt
                if pitch > 0.65:
                    droop_counter += 1
                else:
                    droop_counter = max(0, droop_counter - 1)
                    
                # Distraction looking away
                if abs(yaw - 0.5) > 0.12:
                    distracted_counter += 1
                    distraction_duration += 0.12
                else:
                    distracted_counter = max(0, distracted_counter - 1)
                
                # Composite score
                fatigue_score = (
                    SCORE_WEIGHTS["eye"]  * eye_score +
                    SCORE_WEIGHTS["yawn"] * yawn_score +
                    SCORE_WEIGHTS["cnn"]  * cnn_prob
                )
                
                score_history.append(fatigue_score)
                if len(score_history) > 15:
                    score_history.pop(0)
                avg_score = float(np.mean(score_history))
                
                # Status determination
                if drowsy_counter > 10 or avg_score > 0.65:
                    status = "DROWSY"
                elif droop_counter > 10:
                    status = "HEAD DROOPING"
                elif distracted_counter > 15:
                    status = "DISTRACTED"
                elif drowsy_counter > 5 or avg_score > 0.30:
                    status = "SLIGHTLY DROWSY"
                else:
                    status = "ALERT"
                    
                # Active landmark extraction (Left eye, Right eye, Mouth)
                draw_landmarks = LEFT_EYE + RIGHT_EYE + [13, 14, 78, 308]
                landmarks_list = [
                    {"index": idx, "x": float(landmarks[idx].x), "y": float(landmarks[idx].y)}
                    for idx in draw_landmarks
                ]
            else:
                drowsy_counter = max(0, drowsy_counter - 1)
                distracted_counter = max(0, distracted_counter - 1)
                droop_counter = max(0, droop_counter - 1)
                score_history.append(0.0)
                if len(score_history) > 15:
                    score_history.pop(0)
                avg_score = float(np.mean(score_history))
                status = "NO FACE DETECTED"
                pitch = 0.5
                yaw = 0.5
                
            await websocket.send_json({
                "ear": float(ear),
                "mar": float(mar),
                "cnn_prob": float(cnn_prob),
                "fatigue_score": float(avg_score),
                "status": status,
                "bbox": bbox,
                "landmarks": landmarks_list,
                "use_cnn": USE_CNN,
                "pitch": float(pitch),
                "yaw": float(yaw),
                "blink_count": blink_count,
                "yawn_count": yawn_count,
                "distraction_duration": float(distraction_duration)
            })
            
    except WebSocketDisconnect:
        print("[WS] Client disconnected from WebSocket.")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

# ─── ROOT ENDPOINT ──────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "running",
        "demo_mode": not HAS_REAL_TRACKING,
        "real_tracking_loaded": HAS_REAL_TRACKING,
        "model_loaded": USE_CNN,
        "python_version": os.sys.version,
        "tensorflow_version": tf.__version__ if (HAS_TENSORFLOW and USE_CNN) else "None"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
