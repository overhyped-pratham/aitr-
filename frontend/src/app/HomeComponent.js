"use client";

import React, { useState, useEffect, useRef } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function Home() {
  // Navigation View: "landing" | "dashboard" | "emergency" | "settings"
  const [view, setView] = useState("landing");
  
  // Camera & WebSocket States
  const [cameraActive, setCameraActive] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [useCNN, setUseCNN] = useState(true);
  
  // Real-time Fatigue Metrics
  const [ear, setEar] = useState(0.3);
  const [mar, setMar] = useState(0.0);
  const [cnnProb, setCnnProb] = useState(0.0);
  const [fatigueScore, setFatigueScore] = useState(0.0);
  const [status, setStatus] = useState("ALERT");
  
  // History and Alerts
  const [history, setHistory] = useState([]);
  const [lastAlertTime, setLastAlertTime] = useState(null);
  
  // Gemini Analysis State
  const [analysis, setAnalysis] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [tripDuration, setTripDuration] = useState(45);
  const [timeOfDay, setTimeOfDay] = useState("Night");

  // Map & POI States
  const [poiType, setPoiType] = useState("hospital");
  const [pois, setPois] = useState([]);
  const [selectedPoi, setSelectedPoi] = useState(null);
  const [driverLocation, setDriverLocation] = useState({ lat: 28.6139, lng: 77.2090 }); // Default Delhi
  
  // Emergency Escalation States
  const [emergencyTimer, setEmergencyTimer] = useState(10);
  const [emergencyCountdownActive, setEmergencyCountdownActive] = useState(false);
  const [emergencySent, setEmergencySent] = useState(false);
  const [emergencyPhone, setEmergencyPhone] = useState("+1 (555) 019-2831");
  const [emergencyName, setEmergencyName] = useState("John Doe (Fleet Manager)");
  const [emergencyMuted, setEmergencyMuted] = useState(false);
  const [emergencyHospital, setEmergencyHospital] = useState(null);
  const [emergencyPoliceStation, setEmergencyPoliceStation] = useState(null);
  
  // API Keys (loaded from localStorage or env)
  const [geminiKey, setGeminiKey] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("drowsiness_gemini_key") || "";
    }
    return "";
  });
  const [mapsKey, setMapsKey] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("drowsiness_maps_key") || "";
    }
    return "";
  });

  // Phase 2 State Variables
  const [blinkCount, setBlinkCount] = useState(0);
  const [yawnCount, setYawnCount] = useState(0);
  const [pitch, setPitch] = useState(0.5);
  const [yaw, setYaw] = useState(0.5);
  const [distractionDuration, setDistractionDuration] = useState(0);
  
  const [accidentDetected, setAccidentDetected] = useState(false);
  const [emergencySummary, setEmergencySummary] = useState(null);
  const [emergencySummaryLoading, setEmergencySummaryLoading] = useState(false);
  const [smsResponse, setSmsResponse] = useState(null);
  
  const [routeHistory, setRouteHistory] = useState([{ lat: 28.6139, lng: 77.2090, time: new Date().toLocaleTimeString() }]);
  const [vehicles, setVehicles] = useState([
    { id: "V-109", driver: "Pratham (You)", status: "ALERT", score: 0.0, lat: 28.6139, lng: 77.2090, speed: 65, active: true },
    { id: "V-214", driver: "Sarah Connor (Express)", status: "DISTRACTED", score: 0.45, lat: 28.6250, lng: 77.2200, speed: 78, active: true },
    { id: "V-872", driver: "John Miller (Logistics)", status: "DROWSY", score: 0.88, lat: 28.6010, lng: 77.1980, speed: 45, active: true },
    { id: "V-404", driver: "Mike Peterson (Local)", status: "ALERT", score: 0.05, lat: 28.6300, lng: 77.2500, speed: 52, active: false }
  ]);
  const [selectedVehicleId, setSelectedVehicleId] = useState("V-109");
  
  // Custom emergency contact details for SMS
  const [smsRecipientNumber, setSmsRecipientNumber] = useState("+91 96445 23146");
  const [smsRecipientName, setSmsRecipientName] = useState("Emergency Contact");

  // DOM References
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  const mapContainerRef = useRef(null);
  const leafletMapRef = useRef(null);
  const leafletMarkersRef = useRef([]);
  const authorityMapContainerRef = useRef(null);
  const authorityLeafletMapRef = useRef(null);
  const authorityLeafletMarkersRef = useRef([]);
  const audioCtxRef = useRef(null);
  const alarmIntervalRef = useRef(null);
  const frameIntervalRef = useRef(null);

  // Load API keys on start
  useEffect(() => {
    // Set mock driver coordinates if geolocation is not available
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setDriverLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        },
        () => {
          // Fallback to random coordinates near Delhi or NYC
          setDriverLocation({ lat: 40.7128, lng: -74.0060 });
        }
      );
    }
  }, []);

  // GPS Route Simulation Effect
  useEffect(() => {
    let interval = null;
    if (cameraActive) {
      interval = setInterval(() => {
        setDriverLocation(prev => {
          const nextLat = prev.lat + 0.00012 * (Math.random() * 0.4 + 0.8);
          const nextLng = prev.lng + 0.00018 * (Math.random() * 0.4 + 0.8);
          
          const newPoint = {
            lat: nextLat,
            lng: nextLng,
            time: new Date().toLocaleTimeString()
          };
          
          setRouteHistory(historyPrev => [...historyPrev, newPoint].slice(-35));
          
          setVehicles(vPrev => vPrev.map(v => v.id === "V-109" ? {
            ...v,
            lat: nextLat,
            lng: nextLng
          } : v));
          
          return { lat: nextLat, lng: nextLng };
        });
      }, 3000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [cameraActive]);

  // Sync API Keys with LocalStorage
  const saveApiKeys = (gKey, mKey) => {
    setGeminiKey(gKey);
    setMapsKey(mKey);
    localStorage.setItem("drowsiness_gemini_key", gKey);
    localStorage.setItem("drowsiness_maps_key", mKey);
    alert("API Keys saved successfully!");
    setView("dashboard");
  };

  // ─── AUDIO SYSTEM (WEB AUDIO API) ─────────────────────────
  const startAudioAlarm = () => {
    if (alarmIntervalRef.current || emergencyMuted) return;
    
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioCtxRef.current = new AudioCtx();
      
      alarmIntervalRef.current = setInterval(() => {
        if (!audioCtxRef.current) return;
        const osc = audioCtxRef.current.createOscillator();
        const gain = audioCtxRef.current.createGain();
        osc.connect(gain);
        gain.connect(audioCtxRef.current.destination);
        
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(1800, audioCtxRef.current.currentTime);
        
        gain.gain.setValueAtTime(0.6, audioCtxRef.current.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtxRef.current.currentTime + 0.35);
        
        osc.start();
        osc.stop(audioCtxRef.current.currentTime + 0.4);
      }, 500); // Pulse every 500ms
    } catch (e) {
      console.error("Audio Context failed to start:", e);
    }
  };

  const stopAudioAlarm = () => {
    if (alarmIntervalRef.current) {
      clearInterval(alarmIntervalRef.current);
      alarmIntervalRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
  };

  // Handle manual alarm toggle
  const triggerManualAlarm = () => {
    if (status === "DROWSY") {
      // Mute it
      setEmergencyMuted(true);
      stopAudioAlarm();
    } else {
      setEmergencyMuted(false);
      setStatus("DROWSY");
      setFatigueScore(0.92);
      startAudioAlarm();
      
      const newAlert = {
        time: new Date().toLocaleTimeString(),
        status: "DROWSY (MANUAL)",
        score: 0.92
      };
      setHistory(prev => [newAlert, ...prev].slice(0, 10));
    }
  };

  // ─── WEBCAM & WEBSOCKET ENGINE ────────────────────────────
  function drawLandmarks(landmarks, bbox) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks || landmarks.length === 0) return;

    // Set colors based on status
    let drawColor = "#10b981"; // green
    if (status === "SLIGHTLY DROWSY") drawColor = "#f97316"; // orange
    if (status === "DROWSY") drawColor = "#ef4444"; // red

    // Draw mesh points
    ctx.fillStyle = drawColor;
    landmarks.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw bounding box
    if (bbox && bbox.length === 4) {
      const [x1, y1, x2, y2] = bbox;
      // Map back from 640x480 resolution to canvas resolution
      const scaleX = canvas.width / 640;
      const scaleY = canvas.height / 480;

      ctx.strokeStyle = drawColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(
        x1 * scaleX,
        y1 * scaleY,
        (x2 - x1) * scaleX,
        (y2 - y1) * scaleY
      );
      
      // Draw status label above bounding box
      ctx.fillStyle = drawColor;
      ctx.font = "12px var(--font-mono)";
      ctx.fillText(
        `${status} (${Math.round(fatigueScore * 100)}%)`,
        x1 * scaleX,
        (y1 * scaleY) - 5
      );
    }
  }

  function startFrameStreaming() {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    
    // Offscreen canvas for capturing frames
    offscreenCanvasRef.current = document.createElement("canvas");
    offscreenCanvasRef.current.width = 320; // Downscale to 320x240 to save bandwidth
    offscreenCanvasRef.current.height = 240;
    const ctx = offscreenCanvasRef.current.getContext("2d");

    frameIntervalRef.current = setInterval(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      if (!videoRef.current || videoRef.current.paused) return;

      // Draw video frame to canvas
      ctx.drawImage(videoRef.current, 0, 0, 320, 240);
      
      // Convert to base64 jpeg
      const base64Img = offscreenCanvasRef.current.toDataURL("image/jpeg", 0.65);
      
      // Send payload
      const payload = {
        image: base64Img,
        use_cnn: useCNN
      };
      
      wsRef.current.send(JSON.stringify(payload));
    }, 120); // Sending ~8 frames per second is perfect for real-time
  }

  function connectWebSocket() {
    // Create WebSocket connection to FastAPI
    const wsUrl = BACKEND_URL.replace(/^http/, "ws") + "/ws";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      startFrameStreaming();
    };

    ws.onmessage = (event) => {
      const res = JSON.parse(event.data);
      if (res.error) {
        console.error("Backend Error:", res.error);
        return;
      }

      setEar(res.ear);
      setMar(res.mar);
      setCnnProb(res.cnn_prob);
      setFatigueScore(res.fatigue_score);
      setBlinkCount(res.blink_count || 0);
      setYawnCount(res.yawn_count || 0);
      setPitch(res.pitch || 0.5);
      setYaw(res.yaw || 0.5);
      setDistractionDuration(Math.round(res.distraction_duration || 0));
      
      const oldStatus = status;
      setStatus(res.status);

      // Update local vehicle telemetry in Authority Portal
      setVehicles(prev => prev.map(v => v.id === "V-109" ? {
        ...v,
        status: res.status,
        score: res.fatigue_score
      } : v));

      const isCriticalStatus = res.status === "DROWSY" || res.status === "HEAD DROOPING";
      const wasCriticalStatus = oldStatus === "DROWSY" || oldStatus === "HEAD DROOPING";

      // Handle status change alert logging
      if (isCriticalStatus && !wasCriticalStatus) {
        setLastAlertTime(new Date().toLocaleTimeString());
        
        // Log in alert history
        const newAlert = {
          time: new Date().toLocaleTimeString(),
          status: res.status,
          score: res.fatigue_score
        };
        setHistory(prev => [newAlert, ...prev].slice(0, 10));
        
        // Trigger alarm
        startAudioAlarm();
        
        // Trigger emergency page automatically when critical fatigue reached
        setEmergencyCountdownActive(true);
        setEmergencyTimer(10);
      } else if (!isCriticalStatus && wasCriticalStatus) {
        stopAudioAlarm();
        setEmergencyCountdownActive(false);
      }

      // Draw face mesh landmarks on overlay canvas
      drawLandmarks(res.landmarks, res.bbox);
    };

    ws.onclose = () => {
      setWsConnected(false);
      console.log("WebSocket Disconnected");
    };

    ws.onerror = (err) => {
      console.error("WebSocket Error:", err);
      setWsConnected(false);
    };
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      streamRef.current = stream;
      setCameraActive(true);
      // Note: videoRef.current may be null here because React hasn't re-rendered yet.
      // The useEffect below handles assigning srcObject once the <video> element mounts.
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Could not access camera. Please allow webcam permissions.");
    }
  };

  // Assign stream to video element after it mounts, then connect WebSocket
  useEffect(() => {
    if (cameraActive && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
      // Connect WebSocket only after video is ready
      if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
        connectWebSocket();
      }
    }
  }, [cameraActive]);

  const stopCamera = () => {
    stopAudioAlarm();
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setCameraActive(false);
    setWsConnected(false);
    
    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // ─── GEMINI RISK ANALYSIS ─────────────────────────────────
  const fetchGeminiAnalysis = async () => {
    setAnalysisLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/risk-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fatigueScore: fatigueScore * 100,
          tripDurationMin: tripDuration,
          timeOfDay: timeOfDay,
          customApiKey: geminiKey
        })
      });
      const data = await response.json();
      setAnalysis(data);
    } catch (e) {
      console.error(e);
      alert("Failed to connect to backend for analysis.");
    } finally {
      setAnalysisLoading(false);
    }
  };

  // Update Markers when POIs change
  const updateMapMarkers = async (places) => {
    if (!leafletMapRef.current) return;
    const L = await import("leaflet");
    
    // Clear old markers
    leafletMarkersRef.current.forEach(marker => marker.remove());
    leafletMarkersRef.current = [];
    
    places.forEach(poi => {
      // Custom icon based on type
      let markerColor = "#ef4444"; // hospital red
      if (poiType === "fuel") markerColor = "#f97316"; // gas orange
      if (poiType === "rest_stop") markerColor = "#10b981"; // rest green
      
      const poiIcon = L.divIcon({
        html: `<div style="background-color: ${markerColor}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 8px ${markerColor};"></div>`,
        className: "poi-marker",
        iconSize: [12, 12]
      });

      const marker = L.marker([poi.lat, poi.lng], { icon: poiIcon })
        .addTo(leafletMapRef.current)
        .bindPopup(`<b>${poi.name}</b><br/>Distance: ${poi.distance}`);
        
      leafletMarkersRef.current.push(marker);
    });

    // Fit map to show both driver and first POI
    if (places.length > 0) {
      const bounds = L.latLngBounds([
        [driverLocation.lat, driverLocation.lng],
        [places[0].lat, places[0].lng]
      ]);
      leafletMapRef.current.fitBounds(bounds, { padding: [50, 50] });
    }
  };

  // ─── MAPS & PLACES (LEAFLET + GOOGLE PLACES FALLBACK) ──────
  const fetchNearbyPOIs = async (type) => {
    setPoiType(type);
    try {
      const response = await fetch(`${BACKEND_URL}/api/nearby-assistance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: driverLocation.lat,
          longitude: driverLocation.lng,
          queryType: type,
          customApiKey: mapsKey
        })
      });
      const data = await response.json();
      setPois(data.places || []);
      if (data.places && data.places.length > 0) {
        setSelectedPoi(data.places[0]);
        updateMapMarkers(data.places);
      }
    } catch (e) {
      console.error("POI lookup error:", e);
    }
  };

  // Load Leaflet Map on Emergency Tab Activation
  useEffect(() => {
    if (view === "emergency") {
      // Fetch POIs immediately
      setTimeout(() => {
        fetchNearbyPOIs("hospital");
      }, 0);
      
      // Initialize Leaflet Map
      if (!leafletMapRef.current && typeof window !== "undefined") {
        // Dynamically load leaflet
        const loadLeaflet = async () => {
          // Add Leaflet CSS
          const link = document.createElement("link");
          link.rel = "stylesheet";
          link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
          document.head.appendChild(link);
          
          const L = await import("leaflet");
          
          if (!mapContainerRef.current) return;
          
          const map = L.map(mapContainerRef.current).setView([driverLocation.lat, driverLocation.lng], 14);
          leafletMapRef.current = map;
          
          // Use CartoDB Dark Matter tile layer for premium dark look
          L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
          }).addTo(map);

          // Custom Marker Icon for Driver
          const driverIcon = L.divIcon({
            html: '<div style="background-color: #3b82f6; width: 14px; height: 14px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px #3b82f6;"></div>',
            className: "driver-marker",
            iconSize: [14, 14]
          });
          
          L.marker([driverLocation.lat, driverLocation.lng], { icon: driverIcon })
            .addTo(map)
            .bindPopup("Your Location")
            .openPopup();
        };
        
        loadLeaflet();
      }
    } else {
      // Destroy Leaflet map if leaving view
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
        leafletMarkersRef.current = [];
      }
    }
    
    return () => {
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
      }
    };
  }, [view]);

  const updateAuthorityMarkers = (L) => {
    if (!authorityLeafletMapRef.current) return;
    
    authorityLeafletMarkersRef.current.forEach(m => m.remove());
    authorityLeafletMarkersRef.current = [];
    
    vehicles.forEach(v => {
      let statusColor = "#3b82f6";
      if (v.status === "SLIGHTLY DROWSY") statusColor = "#f97316";
      if (v.status === "DROWSY" || v.status.includes("DROOPING")) statusColor = "#ef4444";
      if (v.status.includes("DISTRACTED")) statusColor = "#a855f7";
      if (v.status.includes("ACCIDENT")) statusColor = "#b91c1c";
      
      const vIcon = L.divIcon({
        html: `<div style="background-color: ${statusColor}; width: 14px; height: 14px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px ${statusColor};"></div>`,
        className: "vehicle-marker",
        iconSize: [14, 14]
      });
      
      const marker = L.marker([v.lat, v.lng], { icon: vIcon })
        .addTo(authorityLeafletMapRef.current)
        .bindPopup(`<b>${v.driver} (${v.id})</b><br/>Status: ${v.status}<br/>Fatigue: ${Math.round(v.score * 100)}%`);
        
      authorityLeafletMarkersRef.current.push(marker);
      
      if (v.id === selectedVehicleId) {
        authorityLeafletMapRef.current.setView([v.lat, v.lng]);
        
        if (v.id === "V-109" && routeHistory.length > 1) {
          const latlngs = routeHistory.map(p => [p.lat, p.lng]);
          const polyline = L.polyline(latlngs, { color: "#3b82f6", weight: 3, opacity: 0.6, dashArray: "5, 5" })
            .addTo(authorityLeafletMapRef.current);
          authorityLeafletMarkersRef.current.push(polyline);
        }
      }
    });
  };

  // Load Leaflet Map on Authority Portal Activation
  useEffect(() => {
    if (view === "authority") {
      if (!authorityLeafletMapRef.current && typeof window !== "undefined") {
        const loadAuthorityLeaflet = async () => {
          if (!document.getElementById("leaflet-css")) {
            const link = document.createElement("link");
            link.id = "leaflet-css";
            link.rel = "stylesheet";
            link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
            document.head.appendChild(link);
          }
          
          const L = await import("leaflet");
          if (!authorityMapContainerRef.current) return;
          
          const map = L.map(authorityMapContainerRef.current).setView([driverLocation.lat, driverLocation.lng], 13);
          authorityLeafletMapRef.current = map;
          
          L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
          }).addTo(map);
          
          updateAuthorityMarkers(L);
        };
        
        loadAuthorityLeaflet();
      } else if (authorityLeafletMapRef.current) {
        import("leaflet").then(L => {
          updateAuthorityMarkers(L);
        });
      }
    } else {
      if (authorityLeafletMapRef.current) {
        authorityLeafletMapRef.current.remove();
        authorityLeafletMapRef.current = null;
        authorityLeafletMarkersRef.current = [];
      }
    }
    
    return () => {
      if (authorityLeafletMapRef.current) {
        authorityLeafletMapRef.current.remove();
        authorityLeafletMapRef.current = null;
      }
    };
  }, [view, vehicles, selectedVehicleId]);

  const selectPoiOnMap = async (poi) => {
    setSelectedPoi(poi);
    if (!leafletMapRef.current) return;
    const L = await import("leaflet");
    leafletMapRef.current.setView([poi.lat, poi.lng], 15);
  };

  const sendEmergencyAlert = async () => {
    setEmergencySent(true);
    setEmergencyCountdownActive(false);
    setEmergencySummaryLoading(true);
    
    try {
      const sosResp = await fetch(`${BACKEND_URL}/api/send-sos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fatigueScore: fatigueScore * 100,
          ear: ear,
          mar: mar,
          blinkCount: blinkCount,
          yawnCount: yawnCount,
          distractionDuration: distractionDuration,
          latitude: driverLocation.lat,
          longitude: driverLocation.lng,
          customApiKey: geminiKey
        })
      });
      const data = await sosResp.json();
      
      setEmergencySummary({ report_text: data.emergency_summary });
      setEmergencyHospital(data.hospital);
      setEmergencyPoliceStation(data.police_station);
      setSmsResponse({ sms_body: data.sos_message, status: data.sms_status, feedback: data.sms_feedback });
      
      // Update fleet vehicle status
      setVehicles(prev => prev.map(v => v.id === "V-109" ? {
        ...v,
        status: status === "ACCIDENT DETECTED" ? "ACCIDENT DETECTED" : "CRITICAL DROWSY",
        score: 1.0
      } : v));
    } catch (e) {
      console.error("SOS alert dispatch error:", e);
    } finally {
      setEmergencySummaryLoading(false);
    }
  };

  // ─── EMERGENCY COUNTDOWN ──────────────────────────────────
  useEffect(() => {
    let interval = null;
    if (emergencyCountdownActive && emergencyTimer > 0) {
      interval = setInterval(() => {
        setEmergencyTimer(t => t - 1);
      }, 1000);
    } else if (emergencyCountdownActive && emergencyTimer === 0 && !emergencySent) {
      setTimeout(() => {
        sendEmergencyAlert();
      }, 0);
    }
    return () => clearInterval(interval);
  }, [emergencyCountdownActive, emergencyTimer]);

  const triggerAccidentSimulation = () => {
    setAccidentDetected(true);
    setFatigueScore(1.0);
    setStatus("ACCIDENT DETECTED");
    setEmergencyCountdownActive(false);
    setEmergencySent(false);
    setEmergencyTimer(0);
    startAudioAlarm();
    
    const newAlert = {
      time: new Date().toLocaleTimeString(),
      status: "ACCIDENT DETECTED (COLLISION IMPACT)",
      score: 1.0
    };
    setHistory(prev => [newAlert, ...prev].slice(0, 10));
    setView("emergency");
    
    // Dispatch alerts immediately
    setTimeout(() => {
      sendEmergencyAlert();
    }, 800);
  };

  const cancelEmergencyAlert = () => {
    setEmergencyCountdownActive(false);
    setEmergencyMuted(true);
    setAccidentDetected(false);
    stopAudioAlarm();
  };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
      
      {/* ─── APP HEADER ──────────────────────────────────────── */}
      <header className="app-header">
        <div className="logo-container" onClick={() => setView("landing")} style={{ cursor: "pointer" }}>
          <span className="logo-icon">🚗</span>
          <span className="logo-text">DriveSafe AI</span>
        </div>
        
        <nav className="nav-links">
          <div className={`nav-item ${view === "landing" ? "active" : ""}`} onClick={() => setView("landing")}>Home</div>
          <div className={`nav-item ${view === "dashboard" ? "active" : ""}`} onClick={() => { setView("dashboard"); if(!cameraActive) startCamera(); }}>Monitor</div>
          <div className={`nav-item ${view === "emergency" ? "active" : ""}`} onClick={() => setView("emergency")}>Safety Hub</div>
          <div className={`nav-item ${view === "authority" ? "active" : ""}`} onClick={() => setView("authority")}>Authority Portal</div>
          <div className={`nav-item ${view === "settings" ? "active" : ""}`} onClick={() => setView("settings")}>Config</div>
        </nav>

        <div className="header-status-indicator">
          <div className={`indicator-dot ${wsConnected ? "connected" : "disconnected"}`}></div>
          <span>{wsConnected ? "MODEL ONLINE" : "OFFLINE DEMO"}</span>
        </div>
      </header>

      {/* ─── APP CONTENT VIEWS ───────────────────────────────── */}
      <main className="view-container">
        
        {/* ─── 1. LANDING PAGE VIEW ──────────────────────────── */}
        {view === "landing" && (
          <div className="landing-view">
            
            {/* Hero Section */}
            <section className="hero-section">
              <div className="hero-tagline">Hackathon Project</div>
              <h1 className="hero-title">DriveSafe AI</h1>
              <h2 className="hero-subtitle" style={{ fontSize: "24px", color: "hsl(var(--accent-blue))", fontWeight: "600" }}>
                AI-Powered Driver Fatigue Prevention System
              </h2>
              <p className="hero-subtitle">
                A real-time driver assistance platform utilizing custom deep learning computer vision to prevent fatigue-related crashes, analyze trip risk using Gemini, and coordinate emergency responses.
              </p>
              <div className="hero-actions">
                <button className="btn btn-primary" onClick={() => { setView("dashboard"); startCamera(); }}>
                  Start Monitoring 🚀
                </button>
                <button className="btn btn-secondary" onClick={() => setView("settings")}>
                  Configure Keys ⚙️
                </button>
              </div>
            </section>

            {/* Problem & Solution Grid */}
            <section className="problem-solution-section">
              <div className="glass-panel problem-card">
                <div className="card-title">🚨 The Problem</div>
                <div className="problem-list">
                  <div className="problem-item">
                    <span className="icon-bullet">🔴</span>
                    <div><b>Fatigue & Microsleep</b>: Drowsy driving is responsible for over 100,000 police-reported crashes annually.</div>
                  </div>
                  <div className="problem-item">
                    <span className="icon-bullet">🔴</span>
                    <div><b>Delayed Interventions</b>: Most warning systems sound a simple alarm after the driver has already closed their eyes, which is often too late.</div>
                  </div>
                  <div className="problem-item">
                    <span className="icon-bullet">🔴</span>
                    <div><b>No Location Context</b>: Standard dashcams do not provide routing assistance to help the driver stop safely.</div>
                  </div>
                </div>
              </div>

              <div className="glass-panel solution-card">
                <div className="card-title">💡 The DriveSafe Solution</div>
                <div className="solution-grid">
                  <div className="solution-item">
                    <span className="icon-bullet">🟢</span>
                    <div><b>Composite Fatigue Index</b>: Evaluates eye closure (EAR), yawning (MAR), and a custom convolutional neural network (CNN) model.</div>
                  </div>
                  <div className="solution-item">
                    <span className="icon-bullet">🟢</span>
                    <div><b>Gemini Risk Analysis</b>: Delivers tailored safety coaching, reaction speed metrics, and stops advice.</div>
                  </div>
                  <div className="solution-item">
                    <span className="icon-bullet">🟢</span>
                    <div><b>Geo-fenced Safestops</b>: Automatically renders the nearest hospitals, food marts, and motels on a dark-themed map in emergency scenarios.</div>
                  </div>
                </div>
              </div>
            </section>

            {/* Team Section */}
            <section className="team-section">
              <h2 style={{ fontSize: "28px", fontWeight: "700" }}>Our Team</h2>
              <div className="team-grid">
                <div className="team-member">
                  <div className="team-avatar">🛡️</div>
                  <div className="team-name">Pratham</div>
                  <div className="team-role">Lead AI & Full Stack Developer</div>
                </div>
                <div className="team-member">
                  <div className="team-avatar">⚡</div>
                  <div className="team-name">Antigravity AI</div>
                  <div className="team-role">AI Coding Partner (Google DeepMind)</div>
                </div>
              </div>
            </section>

          </div>
        )}

        {/* ─── 2. MONITORING DASHBOARD (MAIN MONITOR PAGE) ────── */}
        {view === "dashboard" && (
          <div className="dashboard-view">
            
            {/* Camera Feed Card */}
            <div className="glass-panel camera-card">
              <div className="camera-header">
                <h3>Live Webcam Feed</h3>
                <span style={{ fontSize: "12px", color: wsConnected ? "#10b981" : "#ef4444", fontWeight: "bold" }}>
                  {wsConnected ? "🔴 RUNNING AT 8 FPS" : "⚪ DISCONNECTED"}
                </span>
              </div>

              <div className="camera-wrapper">
                {cameraActive ? (
                  <>
                    <video ref={videoRef} className="webcam-feed" autoPlay playsInline muted></video>
                    <canvas ref={canvasRef} width="640" height="480" className="face-mesh-canvas"></canvas>
                  </>
                ) : (
                  <div className="camera-placeholder">
                    <div className="camera-placeholder-icon">📷</div>
                    <p>Camera feed is inactive.</p>
                    <button className="btn btn-primary" onClick={startCamera}>Enable Webcam</button>
                  </div>
                )}
              </div>

              <div className="camera-actions">
                {cameraActive ? (
                  <button className="btn btn-secondary" onClick={stopCamera}>Pause Stream ⏸️</button>
                ) : (
                  <button className="btn btn-primary" onClick={startCamera}>Start Stream ▶️</button>
                )}
                <button 
                  className={`btn ${status === "DROWSY" ? "btn-danger active" : "btn-danger"}`}
                  onClick={triggerManualAlarm}
                >
                  ⚠️ Simulate Alarm
                </button>
                <button 
                  className="btn btn-danger"
                  onClick={triggerAccidentSimulation}
                  style={{ background: "linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%)", boxShadow: "0 4px 20px rgba(185, 28, 28, 0.4)" }}
                >
                  💥 Simulate Accident
                </button>
                <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "8px", fontSize: "14px" }}>
                  <input 
                    type="checkbox" 
                    id="cnn-check" 
                    checked={useCNN} 
                    onChange={(e) => setUseCNN(e.target.checked)} 
                    style={{ cursor: "pointer" }}
                  />
                  <label htmlFor="cnn-check" style={{ cursor: "pointer" }}>Use CNN Inference Model</label>
                </div>
              </div>
            </div>

            {/* Fatigue Score Gauge Card */}
            <div className="glass-panel fatigue-card">
              <div className="panel-header">
                <h3>Fatigue Assessment</h3>
              </div>

              <div className="fatigue-gauge-container">
                <div className={`fatigue-circle ${
                  status === "DROWSY" || status.includes("DROOPING") ? "critical" : status === "SLIGHTLY DROWSY" || status.includes("DISTRACTED") ? "warning" : "alert"
                }`}>
                  <span className="gauge-score">{Math.round(fatigueScore * 100)}%</span>
                  <span className={`gauge-status ${
                    status === "DROWSY" || status.includes("DROOPING") ? "critical" : status === "SLIGHTLY DROWSY" || status.includes("DISTRACTED") ? "warning" : "alert"
                  }`}>
                    {status}
                  </span>
                </div>
              </div>

              <div className="fatigue-metrics-container">
                <div>
                  <div className="metric-label">
                    <span>👁️ Eye Aspect Ratio (EAR)</span>
                    <span className="metric-value">{ear.toFixed(3)}</span>
                  </div>
                  <div className="metric-bar-outer">
                    <div 
                      className="metric-bar-inner" 
                      style={{ 
                        width: `${Math.min(ear / 0.4, 1.0) * 100}%`,
                        backgroundColor: ear < 0.22 ? "hsl(var(--accent-red))" : "hsl(var(--accent-green))"
                      }}
                    ></div>
                  </div>
                  <span style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px", display: "block" }}>
                    Threshold: &lt; 0.22 (Closed Eye)
                  </span>
                </div>

                <div style={{ marginTop: "8px" }}>
                  <div className="metric-label">
                    <span>👄 Mouth Aspect Ratio (MAR)</span>
                    <span className="metric-value">{mar.toFixed(3)}</span>
                  </div>
                  <div className="metric-bar-outer">
                    <div 
                      className="metric-bar-inner" 
                      style={{ 
                        width: `${Math.min(mar / 1.0, 1.0) * 100}%`,
                        backgroundColor: mar > 0.70 ? "hsl(var(--accent-orange))" : "hsl(var(--accent-green))"
                      }}
                    ></div>
                  </div>
                  <span style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px", display: "block" }}>
                    Threshold: &gt; 0.70 (Yawning)
                  </span>
                </div>

                <div style={{ marginTop: "8px" }}>
                  <div className="metric-label">
                    <span>🤖 Custom CNN Drowsiness Prob</span>
                    <span className="metric-value">{(cnnProb * 100).toFixed(0)}%</span>
                  </div>
                  <div className="metric-bar-outer">
                    <div 
                      className="metric-bar-inner" 
                      style={{ 
                        width: `${cnnProb * 100}%`,
                        backgroundColor: cnnProb > 0.5 ? "hsl(var(--accent-red))" : "hsl(var(--accent-blue))"
                      }}
                    ></div>
                  </div>
                  <span style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px", display: "block" }}>
                    Model: drowsiness_model.keras (128x128 Face Crops)
                  </span>
                </div>
              </div>

              <div style={{ marginTop: "16px", borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "12px" }}>
                <h4 style={{ fontSize: "13px", fontWeight: "600", marginBottom: "8px", color: "#3b82f6" }}>📊 DRIVER SESSION TELEMETRY</h4>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                  <div style={{ padding: "6px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: "8px", display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <span style={{ fontSize: "18px" }}>👁️</span>
                    <span style={{ fontSize: "10px", color: "var(--text-secondary)" }}>Blink Count</span>
                    <span style={{ fontSize: "14px", fontWeight: "bold", color: "#3b82f6" }}>{blinkCount}</span>
                  </div>
                  <div style={{ padding: "6px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: "8px", display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <span style={{ fontSize: "18px" }}>🥱</span>
                    <span style={{ fontSize: "10px", color: "var(--text-secondary)" }}>Yawn Count</span>
                    <span style={{ fontSize: "14px", fontWeight: "bold", color: "#f97316" }}>{yawnCount}</span>
                  </div>
                  <div style={{ padding: "6px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: "8px", display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <span style={{ fontSize: "18px" }}>📐</span>
                    <span style={{ fontSize: "10px", color: "var(--text-secondary)" }}>Head Pitch (Droop)</span>
                    <span style={{ fontSize: "12px", fontWeight: "bold", color: pitch > 0.65 ? "#ef4444" : "#10b981" }}>
                      {pitch.toFixed(2)} {pitch > 0.65 ? "⚠️" : "✓"}
                    </span>
                  </div>
                  <div style={{ padding: "6px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: "8px", display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <span style={{ fontSize: "18px" }}>🔄</span>
                    <span style={{ fontSize: "10px", color: "var(--text-secondary)" }}>Head Yaw (Turn)</span>
                    <span style={{ fontSize: "12px", fontWeight: "bold", color: Math.abs(yaw - 0.5) > 0.12 ? "#a855f7" : "#10b981" }}>
                      {yaw.toFixed(2)} {Math.abs(yaw - 0.5) > 0.12 ? "⚠️" : "✓"}
                    </span>
                  </div>
                </div>
                {distractionDuration > 0 && (
                  <div style={{ marginTop: "8px", padding: "6px 10px", background: "rgba(168, 85, 247, 0.1)", border: "1px solid rgba(168, 85, 247, 0.2)", borderRadius: "8px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "11px" }}>
                    <span style={{ color: "#c084fc", fontWeight: "600" }}>⚠️ Distraction Duration:</span>
                    <span style={{ fontWeight: "bold", color: "#c084fc" }}>{distractionDuration}s</span>
                  </div>
                )}
              </div>
            </div>

            {/* Gemini Analysis Card */}
            <div className="glass-panel analysis-card">
              <div className="panel-header">
                <h3>Gemini-Powered Risk Analysis</h3>
                <div style={{ display: "flex", gap: "10px" }}>
                  <select 
                    className="settings-input" 
                    style={{ padding: "6px", fontSize: "12px" }}
                    value={timeOfDay}
                    onChange={(e) => setTimeOfDay(e.target.value)}
                  >
                    <option value="Morning">Morning</option>
                    <option value="Afternoon">Afternoon</option>
                    <option value="Night">Night</option>
                  </select>
                  <button className="btn btn-primary" style={{ padding: "6px 14px", fontSize: "12px" }} onClick={fetchGeminiAnalysis} disabled={analysisLoading}>
                    {analysisLoading ? "Running AI..." : "Analyze Telemetry 🧠"}
                  </button>
                </div>
              </div>

              <div className="analysis-content">
                {analysis ? (
                  <>
                    <div className="analysis-status">
                      <span>Generated by: {analysis.source}</span>
                      <span>•</span>
                      <span style={{ 
                        color: analysis.severity === "CRITICAL" ? "hsl(var(--accent-red))" :
                               analysis.severity === "HIGH" ? "hsl(var(--accent-orange))" : "hsl(var(--accent-green))",
                        fontWeight: "bold"
                      }}>
                        SEVERITY: {analysis.severity}
                      </span>
                    </div>
                    <p className="analysis-text">{analysis.explanation}</p>
                    <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "8px" }}>
                      <strong>Actionable Recommendations:</strong>
                      {analysis.recommendations && analysis.recommendations.map((rec, idx) => (
                        <div key={idx} className="rec-item">
                          <span className="rec-bullet">👉</span>
                          <div>{rec}</div>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div style={{ color: "var(--text-muted)", textAlign: "center", padding: "30px 0" }}>
                    <p style={{ fontSize: "18px" }}>💡 Ready to Analyze telemetry</p>
                    <p style={{ fontSize: "13px", marginTop: "4px" }}>Click &quot;Analyze Telemetry&quot; above to prompt Gemini with your current EAR, MAR, CNN fatigue score, trip time, and time of day.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Alert History Card */}
            <div className="glass-panel history-card">
              <div className="panel-header">
                <h3>Alert History Log</h3>
                <button 
                  className="btn btn-secondary" 
                  style={{ padding: "4px 10px", fontSize: "11px" }}
                  onClick={() => setHistory([])}
                >
                  Clear Logs
                </button>
              </div>

              <div className="history-list">
                {history.length > 0 ? (
                  history.map((item, idx) => (
                    <div key={idx} className={`history-row ${item.status.includes("DROWSY") ? "drowsy" : "warning"}`}>
                      <span className="history-type">
                        🚨 {item.status} - Fatigue Score: {Math.round(item.score * 100)}%
                      </span>
                      <span className="history-time">{item.time}</span>
                    </div>
                  ))
                ) : (
                  <div style={{ color: "var(--text-muted)", padding: "16px 0", textAlign: "center", fontSize: "13px" }}>
                    No drowsiness alerts recorded in this session.
                  </div>
                )}
              </div>
            </div>

          </div>
        )}

        {/* ─── 3. EMERGENCY & MAPS VIEW ───────────────────────── */}
        {view === "emergency" && (
          <div className="emergency-view">
            
            {/* Interactive Map Card */}
            <div className="glass-panel map-card">
              <div className="panel-header">
                <h3>Active Safestops Nav</h3>
                <span style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                  Location: {driverLocation.lat.toFixed(4)}, {driverLocation.lng.toFixed(4)}
                </span>
              </div>

              <div className="map-outer">
                <div ref={mapContainerRef} className="google-map-container"></div>
                {!mapsKey && (
                  <div className="map-fallback-overlay" style={{ pointerEvents: "none" }}>
                    <div style={{ background: "rgba(11, 15, 25, 0.9)", border: "1px solid rgba(255, 255, 255, 0.08)", padding: "20px", borderRadius: "12px", pointerEvents: "auto" }}>
                      <h4>Google Maps API Key Missing</h4>
                      <p className="map-instructions" style={{ marginTop: "6px" }}>
                        We are displaying an interactive OpenStreetMap (Leaflet) fallback. Provide a Google Maps key in the Config tab to enable full Google Maps API integration.
                      </p>
                      <button className="btn btn-secondary" style={{ padding: "6px 14px", fontSize: "12px" }} onClick={() => setView("settings")}>
                        Configure Maps Key ⚙️
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* POI Panel & Contact Panel */}
            <div className="poi-panel">
              <div className="glass-panel" style={{ flex: 1, display: "flex", flexDirection: "column", gap: "16px" }}>
                
                <div className="poi-tabs">
                  <div 
                    className={`poi-tab-btn ${poiType === "hospital" ? "active" : ""}`}
                    onClick={() => fetchNearbyPOIs("hospital")}
                  >
                    🏥 Hospital Finder
                  </div>
                  <div 
                    className={`poi-tab-btn ${poiType === "police" ? "active" : ""}`}
                    onClick={() => fetchNearbyPOIs("police")}
                  >
                    🛡️ Police Finder
                  </div>
                  <div 
                    className={`poi-tab-btn ${poiType === "rest_stop" ? "active" : ""}`}
                    onClick={() => fetchNearbyPOIs("rest_stop")}
                  >
                    🏨 Rest Areas
                  </div>
                </div>

                <div className="poi-list" style={{ maxHeight: "150px" }}>
                  {pois.length > 0 ? (
                    pois.map((poi, idx) => (
                      <div 
                        key={idx} 
                        className={`poi-card ${selectedPoi?.name === poi.name ? "selected" : ""}`}
                        onClick={() => selectPoiOnMap(poi)}
                        style={{ padding: "8px" }}
                      >
                        <div className="poi-name" style={{ fontSize: "13px" }}>{poi.name}</div>
                        <div className="poi-address" style={{ fontSize: "11px" }}>{poi.address}</div>
                      </div>
                    ))
                  ) : (
                    <div style={{ color: "var(--text-muted)", padding: "20px 0", textAlign: "center", fontSize: "12px" }}>
                      No nearby services found.
                    </div>
                  )}
                </div>

              </div>

              {/* Module 2: SOS Emergency Action System */}
              <div className="glass-panel escalation-panel" style={{ background: "rgba(239, 68, 68, 0.04)", border: "1px solid rgba(239, 68, 68, 0.15)" }}>
                <div className="escalation-title">
                  <span>🚨 SOS Emergency System</span>
                </div>
                
                {emergencyCountdownActive && !emergencySent && (
                  <div className="escalation-timer" style={{ fontSize: "20px", padding: "6px" }}>
                    AUTOMATIC SOS DISPATCH IN: {emergencyTimer}s
                  </div>
                )}
                {emergencySent && (
                  <div className="escalation-timer" style={{ color: "#10b981", background: "rgba(16, 185, 129, 0.1)", borderColor: "rgba(16, 185, 129, 0.3)", fontSize: "14px", padding: "6px" }}>
                    ✓ SOS DISPATCHED TO EMERGENCY CONTACTS
                  </div>
                )}

                <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "12px", color: "var(--text-secondary)" }}>
                  <div><b>Emergency Contact 1:</b> {emergencyName} ({smsRecipientNumber})</div>
                  <div><b>Emergency Contact 2:</b> Highway Rescue Dispatch (+91 99999 11222)</div>
                </div>

                <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
                  <button className="btn btn-danger" style={{ flex: 1, padding: "10px", fontSize: "13px" }} onClick={sendEmergencyAlert} disabled={emergencySent}>
                    Send SOS Now 📲
                  </button>
                  {emergencyCountdownActive && !emergencySent && (
                    <button className="btn btn-secondary" style={{ padding: "10px", fontSize: "13px" }} onClick={cancelEmergencyAlert}>
                      Cancel 🛑
                    </button>
                  )}
                </div>

                {/* Call Emergency Numbers */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginTop: "4px" }}>
                  <button className="btn btn-danger" style={{ padding: "10px", fontSize: "12px", background: "#b91c1c" }} onClick={() => window.open("tel:112")}>
                    📞 Call 112 (Police)
                  </button>
                  <button className="btn btn-danger" style={{ padding: "10px", fontSize: "12px", background: "linear-gradient(135deg, #f97316 0%, #d97706 100%)" }} onClick={() => window.open("tel:108")}>
                    🚑 Call 108 (Ambulance)
                  </button>
                </div>
              </div>

            </div>

            {/* Right Column / Side Columns: Gemini report and Nearest Cards */}
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              
              {/* Module 1: Gemini Emergency Report */}
              <div className="glass-panel">
                <h3 style={{ fontSize: "14px", fontWeight: "700", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "6px", color: "hsl(var(--accent-red))" }}>
                  🧠 Gemini AI Emergency Report
                </h3>
                {emergencySummaryLoading ? (
                  <p style={{ color: "var(--text-muted)", fontSize: "12px", padding: "10px 0" }}>Generating emergency dispatch analysis...</p>
                ) : emergencySummary ? (
                  <div style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.4", marginTop: "6px" }}>
                    <p style={{ background: "rgba(255,255,255,0.03)", padding: "10px", borderRadius: "8px", borderLeft: "4px solid #ef4444" }}>
                      {emergencySummary.report_text || emergencySummary.summary}
                    </p>
                  </div>
                ) : (
                  <p style={{ color: "var(--text-muted)", fontSize: "12px", padding: "10px 0" }}>Waiting for emergency trigger to generate report...</p>
                )}
              </div>

              {/* Module 2: Nearest Hospital Card */}
              <div className="glass-panel">
                <h3 style={{ fontSize: "14px", fontWeight: "700", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "6px", color: "#3b82f6" }}>
                  🏥 Nearest Emergency Hospital
                </h3>
                {emergencyHospital ? (
                  <div className="poi-card" style={{ marginTop: "8px", padding: "10px" }} onClick={() => window.open(emergencyHospital.maps_link, "_blank")}>
                    <div className="poi-name" style={{ fontSize: "13px" }}>{emergencyHospital.name}</div>
                    <div className="poi-address" style={{ fontSize: "11px" }}>{emergencyHospital.address}</div>
                    <div className="poi-details" style={{ fontSize: "11px", marginTop: "4px" }}>
                      <span>📏 Distance: {emergencyHospital.distance}</span>
                      <span style={{ color: "#3b82f6", fontWeight: "bold" }}>Open Maps 🧭</span>
                    </div>
                  </div>
                ) : (
                  <p style={{ color: "var(--text-muted)", fontSize: "12px", padding: "10px 0" }}>Searching nearest hospital...</p>
                )}
              </div>

              {/* Module 2: Nearest Police Station Card */}
              <div className="glass-panel">
                <h3 style={{ fontSize: "14px", fontWeight: "700", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "6px", color: "#a855f7" }}>
                  🛡️ Nearest Police Station
                </h3>
                {emergencyPoliceStation ? (
                  <div className="poi-card" style={{ marginTop: "8px", padding: "10px" }} onClick={() => window.open(emergencyPoliceStation.maps_link, "_blank")}>
                    <div className="poi-name" style={{ fontSize: "13px" }}>{emergencyPoliceStation.name}</div>
                    <div className="poi-address" style={{ fontSize: "11px" }}>{emergencyPoliceStation.address}</div>
                    <div className="poi-details" style={{ fontSize: "11px", marginTop: "4px" }}>
                      <span>📏 Distance: {emergencyPoliceStation.distance}</span>
                      <span style={{ color: "#a855f7", fontWeight: "bold" }}>Open Maps 🧭</span>
                    </div>
                  </div>
                ) : (
                  <p style={{ color: "var(--text-muted)", fontSize: "12px", padding: "10px 0" }}>Searching nearest police station...</p>
                )}
              </div>

            </div>

          </div>
        )}

        {/* ─── 3.5 AUTHORITY PORTAL VIEW ───────────────────────── */}
        {view === "authority" && (
          <div className="authority-view" style={{ display: "grid", gridTemplateColumns: "1.2fr 2fr", gap: "20px", minHeight: "calc(100vh - 180px)" }}>
            
            {/* Left Column: Active Fleet Monitoring & Reports */}
            <div style={{ display: "flex", flexDirection: "column", gap: "16px", overflowY: "auto", paddingRight: "4px" }}>
              
              {/* Fleet List */}
              <div className="glass-panel" style={{ padding: "16px" }}>
                <h3 style={{ fontSize: "15px", fontWeight: "700", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "8px", color: "#3b82f6" }}>
                  🚙 Active Fleet Vehicles ({vehicles.length})
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                  {vehicles.map(v => {
                    const isSelected = selectedVehicleId === v.id;
                    let dotColor = "#10b981";
                    if (v.status === "SLIGHTLY DROWSY") dotColor = "#f97316";
                    if (v.status.includes("DROWSY") || v.status.includes("DROOPING")) dotColor = "#ef4444";
                    if (v.status.includes("DISTRACTED")) dotColor = "#a855f7";
                    if (v.status.includes("ACCIDENT")) dotColor = "#b91c1c";
                    
                    return (
                      <div 
                        key={v.id}
                        className={`poi-card ${isSelected ? "selected" : ""}`}
                        style={{ cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px", borderRadius: "8px" }}
                        onClick={() => setSelectedVehicleId(v.id)}
                      >
                        <div>
                          <div style={{ fontWeight: "bold", fontSize: "13px", color: "var(--text-primary)" }}>{v.driver}</div>
                          <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                            ID: {v.id} • Lat: {v.lat.toFixed(4)}, Lng: {v.lng.toFixed(4)}
                          </div>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "4px" }}>
                          <span style={{ fontSize: "11px", fontWeight: "bold", color: dotColor, display: "flex", alignItems: "center", gap: "4px" }}>
                            <span style={{ display: "inline-block", width: "8px", height: "8px", borderRadius: "50%", backgroundColor: dotColor }}></span>
                            {v.status}
                          </span>
                          <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                            Fatigue: {Math.round(v.score * 100)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Selected Vehicle Telemetry */}
              {(() => {
                const sv = vehicles.find(v => v.id === selectedVehicleId);
                if (!sv) return null;
                return (
                  <div className="glass-panel" style={{ padding: "16px" }}>
                    <h3 style={{ fontSize: "15px", fontWeight: "700", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "8px", color: "#3b82f6" }}>
                      📡 Telemetry & Control Panel: {sv.driver}
                    </h3>
                    
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", fontSize: "13px" }}>
                      <div><b>Vehicle Speed:</b> {sv.speed} km/h</div>
                      <div><b>Active Connection:</b> {sv.active ? "✅ YES" : "❌ NO"}</div>
                      <div><b>Latitude:</b> {sv.lat.toFixed(6)}</div>
                      <div><b>Longitude:</b> {sv.lng.toFixed(6)}</div>
                    </div>

                    <div style={{ marginTop: "14px", display: "flex", flexDirection: "column", gap: "8px" }}>
                      <button 
                        className="btn btn-secondary" 
                        style={{ padding: "8px", fontSize: "12px" }}
                        onClick={() => alert(`Siren warning alert transmitted to ${sv.driver}'s dashboard.`)}
                      >
                        🔊 Remote Siren Warning
                      </button>
                      <button 
                        className="btn btn-secondary" 
                        style={{ padding: "8px", fontSize: "12px", color: "#10b981", borderColor: "rgba(16, 185, 129, 0.2)" }}
                        onClick={() => alert(`Calling driver ${sv.driver} at emergency contact number.`)}
                      >
                        📞 Call Driver Hotline
                      </button>
                      <button 
                        className="btn btn-danger" 
                        style={{ padding: "8px", fontSize: "12px" }}
                        onClick={() => alert(`Emergency rescue team dispatched to Lat ${sv.lat.toFixed(6)}, Lng ${sv.lng.toFixed(6)}`)}
                      >
                        🚑 Dispatch Highway Rescue
                      </button>
                    </div>
                  </div>
                );
              })()}

              {/* SMS Dispatch log */}
              {selectedVehicleId === "V-109" && (smsResponse || emergencySent) && (
                <div className="glass-panel" style={{ padding: "16px" }}>
                  <h3 style={{ fontSize: "15px", fontWeight: "700", marginBottom: "10px", color: "#10b981", display: "flex", alignItems: "center", gap: "6px" }}>
                    📲 SMS Dispatch Logs (Twilio Portal)
                  </h3>
                  <div style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", padding: "12px", fontSize: "12px", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                    <div><b>Status:</b> <span style={{ color: "#10b981" }}>{smsResponse?.status ? smsResponse.status.toUpperCase() : "DISPATCHED"}</span></div>
                    {smsResponse?.sid && <div><b>Sid:</b> {smsResponse.sid}</div>}
                    <div><b>To Contact:</b> {smsRecipientName} ({smsRecipientNumber})</div>
                    <div style={{ marginTop: "6px", borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: "6px", color: "var(--text-primary)" }}>
                      <b>SMS Text:</b>
                      <pre style={{ whiteSpace: "pre-wrap", marginTop: "4px", fontSize: "11px", fontFamily: "var(--font-mono)" }}>
                        {smsResponse?.sms_body || `EMERGENCY ALERT: Driver may be unconscious.\nLocation: Lat ${driverLocation.lat.toFixed(6)}, Lng ${driverLocation.lng.toFixed(6)}\nGoogle Maps: https://maps.google.com/?q=${driverLocation.lat},${driverLocation.lng}`}
                      </pre>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column: Live Map tracking & Gemini Report */}
            <div style={{ display: "flex", flexDirection: "column", gap: "16px", height: "100%" }}>
              
              {/* Map Tracker */}
              <div className="glass-panel" style={{ flex: 1, display: "flex", flexDirection: "column", padding: "16px" }}>
                <h3 style={{ fontSize: "15px", fontWeight: "700", marginBottom: "10px" }}>
                  🗺️ Dispatch Map Tracker: {vehicles.find(v => v.id === selectedVehicleId)?.driver}
                </h3>
                <div style={{ flex: 1, position: "relative", minHeight: "300px" }}>
                  <div ref={authorityMapContainerRef} style={{ width: "100%", height: "100%", borderRadius: "8px", minHeight: "300px" }}></div>
                </div>
              </div>

              {/* Gemini incident summary */}
              {selectedVehicleId === "V-109" && (emergencySummary || emergencySummaryLoading) && (
                <div className="glass-panel" style={{ padding: "16px" }}>
                  <h3 style={{ fontSize: "15px", fontWeight: "700", marginBottom: "12px", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "8px", color: "hsl(var(--accent-red))", display: "flex", alignItems: "center", gap: "8px" }}>
                    🧠 Gemini Emergency Rescue Summary
                  </h3>
                  
                  {emergencySummaryLoading ? (
                    <div style={{ color: "var(--text-muted)", padding: "10px 0", fontSize: "13px" }}>
                      Generating critical dispatch coordinates and rescue plan via Gemini...
                    </div>
                  ) : (
                    <div style={{ fontSize: "13px", display: "flex", flexDirection: "column", gap: "10px" }}>
                      <div>
                        <span style={{ fontSize: "11px", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: "bold" }}>Source: {emergencySummary.source}</span>
                      </div>
                      <div>
                        <b>⚠️ Incident Hazard:</b>
                        <p style={{ marginTop: "4px", color: "var(--text-secondary)", lineHeight: "1.4" }}>{emergencySummary.incident_hazard}</p>
                      </div>
                      <div>
                        <b>📊 Telemetry Status:</b>
                        <p style={{ marginTop: "4px", color: "var(--text-secondary)", lineHeight: "1.4" }}>{emergencySummary.telemetry_status}</p>
                      </div>
                      <div style={{ background: "rgba(239, 68, 68, 0.05)", border: "1px solid rgba(239, 68, 68, 0.15)", padding: "10px", borderRadius: "8px" }}>
                        <b style={{ color: "hsl(var(--accent-red))" }}>🚑 Recommended Rescue Action:</b>
                        <p style={{ marginTop: "4px", color: "var(--text-primary)", lineHeight: "1.4" }}>{emergencySummary.recommended_action}</p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

          </div>
        )}

        {/* ─── 4. SETTINGS VIEW (API CONFIG) ─────────────────── */}
        {view === "settings" && (
          <div className="glass-panel settings-view">
            <h2 style={{ marginBottom: "8px", borderBottom: "1px solid rgba(255, 255, 255, 0.1)", paddingBottom: "12px" }}>
              🔑 API Keys Configuration
            </h2>
            <p style={{ fontSize: "14px", color: "var(--text-secondary)", lineHeight: "1.5" }}>
              DriveSafe AI can run completely offline in demo mode. However, to unlock its full potential, you can configure your API credentials. Credentials are saved locally in your browser cache.
            </p>

            <div className="settings-group">
              <label className="settings-label">Gemini API Key</label>
              <input 
                type="password" 
                className="settings-input" 
                placeholder="Enter Gemini API Key (e.g. AIzaSy...)" 
                value={geminiKey}
                onChange={(e) => setGeminiKey(e.target.value)}
              />
              <span className="settings-hint">
                Required for real-time generative safety analysis.
              </span>
            </div>

            <div className="settings-group">
              <label className="settings-label">Google Maps API Key</label>
              <input 
                type="password" 
                className="settings-input" 
                placeholder="Enter Google Maps API Key (e.g. AIzaSy...)" 
                value={mapsKey}
                onChange={(e) => setMapsKey(e.target.value)}
              />
              <span className="settings-hint">
                Used for Google Map renders. Defaults to OpenStreetMap if empty.
              </span>
            </div>

            <div style={{ display: "flex", gap: "12px", marginTop: "16px" }}>
              <button 
                className="btn btn-primary" 
                onClick={() => saveApiKeys(geminiKey, mapsKey)}
                style={{ flex: 1 }}
              >
                Save Configurations
              </button>
              <button 
                className="btn btn-secondary" 
                onClick={() => {
                  setGeminiKey("");
                  setMapsKey("");
                  localStorage.removeItem("drowsiness_gemini_key");
                  localStorage.removeItem("drowsiness_maps_key");
                  alert("Configurations cleared!");
                }}
              >
                Reset
              </button>
            </div>
          </div>
        )}

      </main>

      {/* ─── CRITICAL OVERLAY WARNING ALERT ─────────────────── */}
      {status === "DROWSY" && !emergencyMuted && (
        <div className="critical-overlay-alert">
          <div className="critical-alert-circle">⚠️</div>
          <h1 className="critical-alert-title">CRITICAL FATIGUE DETECTED!</h1>
          <p className="critical-alert-subtitle">
            DriveSafe AI has detected severe signs of drowsiness or closed eyes. Please wake up and pull over safely immediately.
          </p>
          <div style={{ display: "flex", gap: "16px", marginTop: "16px" }}>
            <button className="btn btn-primary" style={{ fontSize: "16px", padding: "16px 32px" }} onClick={() => { setView("emergency"); setEmergencyMuted(true); stopAudioAlarm(); }}>
              🏥 Navigate to Hospital / Rest Stop
            </button>
            <button className="btn btn-secondary" style={{ fontSize: "16px", padding: "16px 32px", background: "rgba(255,255,255,0.1)" }} onClick={cancelEmergencyAlert}>
              Dismiss Alarm 🛑
            </button>
          </div>
        </div>
      )}

      {/* ─── APP FOOTER ─────────────────────────────────────── */}
      <footer className="app-footer">
        <p>© 2026 DriveSafe AI. Designed for Hackathon Driver Safety. Powered by TensorFlow & Gemini.</p>
      </footer>

    </div>
  );
}
