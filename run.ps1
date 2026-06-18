# DriveSafe AI — Launch Script
# Run this script in PowerShell to launch both the Backend and Frontend.

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   🚗 DRIVESAFE AI LAUNCHER" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Ensure we are in the workspace root
$WorkspaceRoot = Get-Location
Write-Host "📍 Workspace Root: $WorkspaceRoot" -ForegroundColor Yellow

# 1. Start FastAPI Backend in background
Write-Host "🔄 Starting FastAPI Python Backend..." -ForegroundColor Cyan
if (Test-Path "backend\venv\Scripts\python.exe") {
    $BackendProcess = Start-Process -FilePath "backend\venv\Scripts\python.exe" -ArgumentList "-m uvicorn main:app --host 127.0.0.1 --port 8000" -WorkingDirectory "backend" -NoNewWindow -PassThru
    Write-Host "✅ Backend started in background (PID: $($BackendProcess.Id))" -ForegroundColor Green
} else {
    Write-Error "❌ Python virtual environment or python.exe not found at 'backend\venv\Scripts\python.exe'."
    Exit
}

# Wait for backend to spin up
Write-Host "⏳ Waiting 3 seconds for backend server to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# 2. Start Next.js Frontend in background
Write-Host "🚀 Starting Next.js Dev Server..." -ForegroundColor Cyan
if (Test-Path "frontend\package.json") {
    $FrontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -WorkingDirectory "frontend" -NoNewWindow -PassThru
    Write-Host "✅ Next.js dev server started in background (PID: $($FrontendProcess.Id))" -ForegroundColor Green
} else {
    Write-Error "❌ Next.js folder not found at 'frontend'."
    Exit
}

Write-Host "⏳ Waiting 3 seconds for frontend compilation..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# 3. Open Web Browser
Write-Host "🌐 Launching DriveSafe AI in your default browser..." -ForegroundColor Green
Start-Process "http://localhost:3000"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "🎉 Platform is active!" -ForegroundColor Green
Write-Host "👉 Dashboard: http://localhost:3000" -ForegroundColor Green
Write-Host "👉 Backend API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C in this terminal to exit any processes or manage manually." -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Cyan
