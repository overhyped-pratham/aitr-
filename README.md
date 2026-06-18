# 🚗 SafeDrive AI — Driver Drowsiness Detection & SOS Dispatch System

SafeDrive AI is an intelligent driver safety monitoring system that combines real-time computer vision drowsiness detection with an automatic SOS emergency dispatch pipeline. When critical fatigue or an accident is detected, the system automatically escalates alerts, generates an AI-powered telemetry emergency report, and broadcasts an SOS SMS with live GPS coordinates, Google Maps links, and nearest emergency service locations to dispatch operators.

---

## 🌟 Key Features

### 1. Multi-Stage SOS Escalation Pipeline
* **Normal Phase (0 - 10s)**: Standard periodic alert alarms when minor drowsiness is detected.
* **Warning Phase (10s - 20s)**: High-frequency alarm beeps with a centered red visual countdown warning overlay on the screen (dismissible by pressing the `SPACEBAR`).
* **Critical Phase (> 20s)**: Automatic trigger of the SOS rescue system:
  1. **GPS Geolocation**: Resolves the driver's approximate lat/long coordinates.
  2. **AI Safety Report**: Calls the Gemini API to analyze the driver's telemetric state and generate a concise emergency rescue summary.
  3. **Emergency Places Lookup**: Queries Google Places to find the closest hospitals and police stations.
  4. **Emergency SMS Dispatch**: Broadcasts a formatted distress message with live maps links and nearest service contact info via Twilio.

### 2. Next.js Web Frontend Dashboard
* **Driver View**: A glassmorphic dark-mode HUD dashboard with live webcam streaming, EAR/MAR meters, yawn & blink counters, head tilt (pitch/yaw) telemetry, and active warnings.
* **Safety Hub & SOS Info**: Real-time display of Gemini-generated telemetry reports.
* **Authority Portal / Fleet View**: Allows dispatch operators to monitor active vehicle fleets, view safety logs, see active coordinates on an interactive Map, and trigger manual simulations.

### 3. FastAPI Python Backend Server
* **WebSocket Server**: High-throughput frame receiver that processes up to ~8-10 frames per second using MediaPipe Face Mesh.
* **REST Endpoints**: Includes endpoints for risk analysis, nearby emergency facility lookup, AI rescue summaries, and Twilio SMS dispatch.
* **Graceful Fallbacks**: Automatically falls back to rule-based EAR/MAR detection if TensorFlow is not installed, and utilizes an AI fallback summary engine if the Gemini API hits rate limits.

---

## 📂 Project Structure

```
├── .env.example                    # Template for environment credentials
├── README.md                       # Comprehensive project documentation
├── drowsiness_model.keras          # Custom trained CNN model
├── realtime_detection.py           # Standalone local Python detection script
├── run.ps1                         # PowerShell script to launch both servers
│
├── backend/                        # FastAPI Python Backend
│   ├── main.py                     # WebSocket & REST API endpoints
│   ├── requirements.txt            # Python dependencies
│   └── venv/                       # Virtual environment
│
├── frontend/                       # Next.js React Web App
│   ├── src/app/
│   │   ├── page.js                 # App Entry point
│   │   ├── HomeComponent.js       # Main dashboard layout and logic
│   │   └── globals.css             # Glassmorphism styling tokens
│   └── package.json                # NPM packages
```

---

## ⚙️ Configuration (.env)

Create a `.env` file in the root workspace directory containing the following credentials:

```ini
# Gemini API Key (For AI telemetry summary analysis)
GEMINI_API_KEY=your_gemini_api_key

# Google Maps API Key (Optional; fallbacks to Leaflet OSM if not set)
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_google_maps_key

# Twilio SMS Recipient (Destination emergency contact)
TWILIO_TO_NUMBER=+91xxxxxxxxxx
TWILIO_TO_NUMBER_2=+91xxxxxxxxxx  # Optional secondary recipient

# Twilio Sending Credentials
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_MESSAGING_SERVICE_SID=your_messaging_service_sid  # Or use TWILIO_PHONE_NUMBER
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
```

---

## 🚀 Getting Started

### Option A: Run the Web Platform (Frontend + Backend)

Using the launcher script in PowerShell:
```powershell
./run.ps1
```
This automatically starts:
1. The FastAPI Python server at [http://127.0.0.1:8000](http://127.0.0.1:8000)
2. The Next.js dev server at [http://localhost:3000](http://localhost:3000)
3. Opens your default web browser to the dashboard.

*To start them manually:*
* **Backend**: `cd backend && venv\Scripts\python.exe -m uvicorn main:app --reload`
* **Frontend**: `cd frontend && npm run dev`

---

### Option B: Run the Standalone Local Script

To run the camera detection window directly in your command line:

```cmd
d:
cd d:\hackathon1
backend\venv\Scripts\python.exe realtime_detection.py
```

#### Controls:
* **`SPACEBAR`**: Dismisses/cancels active warnings and resets the SOS timer.
* **`Q`**: Safely quits the camera feed and exits.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend**: Next.js (React), Leaflet (Maps), CSS (Glassmorphic Dark UI)
* **Backend**: FastAPI (Python), Uvicorn, MediaPipe Face Mesh, Python-dotenv
* **SMS & Communications**: Twilio REST API
* **Generative AI**: Google Gemini 2.5 Flash API
* **Sound Alerts**: Pygame & Winsound (automatic background-thread fallback on Windows)
