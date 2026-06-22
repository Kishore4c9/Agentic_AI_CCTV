@echo off
REM ==========================================================================
REM Agentic AI CCTV Monitoring Framework — Module Launcher (Windows)
REM ==========================================================================
REM Launches individual modules in separate terminal windows for development
REM and debugging. Each module runs independently.
REM
REM Usage:
REM   scripts\launch-modules.bat
REM ==========================================================================

setlocal

set CONFIG_FILE=config.yaml

echo ============================================================
echo  Module Launcher — Individual Services (Windows)
echo ============================================================
echo.

REM -- 1. Mosquitto MQTT Broker --------------------------------------------
echo [1/7] Mosquitto MQTT Broker
start "MQTT Broker" cmd /k "echo Starting Mosquitto MQTT Broker... && mosquitto -v"
timeout /t 2 /nobreak >nul

REM -- 2. MQTT Event Monitor (subscribe to all events) ---------------------
echo [2/7] MQTT Event Monitor
start "MQTT Monitor" cmd /k "echo Monitoring MQTT events... && mosquitto_sub -t +/+/+/# -v"

REM -- 3. Detection Pipeline (cameras + detection + tracking) --------------
echo [3/7] Detection Pipeline
start "Detection Pipeline" cmd /k "echo Starting detection pipeline... && python -c \"import asyncio; from agentic_cctv.main import run_application; asyncio.run(run_application('%CONFIG_FILE%'))\""

REM -- 4. Health API Server ------------------------------------------------
echo [4/7] Health API (embedded in main app, port 8080)
echo       Access: http://localhost:8080/api/health/devices

REM -- 5. Dashboard Server -------------------------------------------------
echo [5/7] Dashboard (embedded in main app, port 8081)
echo       Access: http://localhost:8081/dashboard

REM -- 6. Mobile API Server ------------------------------------------------
echo [6/7] Mobile API (embedded in main app, port 8082)
echo       Access: http://localhost:8082/api/mobile

REM -- 7. Test Runner (watch mode) -----------------------------------------
echo [7/7] Test Runner
start "Test Runner" cmd /k "echo Running tests... && python -m pytest tests/ -q --tb=short"

echo.
echo ============================================================
echo  All modules launched. Close individual windows to stop.
echo ============================================================
echo.
pause
