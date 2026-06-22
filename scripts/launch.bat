@echo off
REM ==========================================================================
REM Agentic AI CCTV Monitoring Framework — Windows Launcher
REM ==========================================================================
REM Starts all services: Mosquitto MQTT broker and the main application.
REM Each service runs in its own terminal window.
REM
REM Usage:
REM   scripts\launch.bat                  (uses config.yaml)
REM   scripts\launch.bat myconfig.yaml    (uses custom config)
REM ==========================================================================

setlocal

set CONFIG_FILE=%1
if "%CONFIG_FILE%"=="" set CONFIG_FILE=config.yaml

echo ============================================================
echo  Agentic AI CCTV Monitoring Framework — Launcher (Windows)
echo ============================================================
echo.
echo Config file: %CONFIG_FILE%
echo.

REM -- Check Python is available -------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

REM -- Check config file exists --------------------------------------------
if not exist "%CONFIG_FILE%" (
    echo [WARN] Config file '%CONFIG_FILE%' not found.
    echo        Generating default config from config.example.yaml...
    if exist config.example.yaml (
        copy config.example.yaml "%CONFIG_FILE%" >nul
        echo        Created %CONFIG_FILE% — edit it before running again.
    ) else (
        echo        No config.example.yaml found either. The app will generate defaults.
    )
)

REM -- Start Mosquitto MQTT Broker -----------------------------------------
echo [1/5] Starting Mosquitto MQTT broker...
where mosquitto >nul 2>&1
if errorlevel 1 (
    echo [WARN] Mosquitto not found in PATH. Skipping broker launch.
    echo        Install from https://mosquitto.org/download/
    echo        Or start it manually: mosquitto
) else (
    start "MQTT Broker — Mosquitto" cmd /k "mosquitto -v"
    timeout /t 2 /nobreak >nul
    echo       Mosquitto started in a new window.
)

REM -- Start Health API Server (port 8080) ---------------------------------
echo [2/5] Starting Health API server (port 8080)...
start "Health API — Port 8080" cmd /k "python -m agentic_cctv.main %CONFIG_FILE%"
REM Note: Health API is embedded in the main application

REM -- Start Dashboard Server (port 8081) ----------------------------------
echo [3/5] Dashboard will be available at http://localhost:8081/dashboard

REM -- Start Mobile API Server (port 8082) ---------------------------------
echo [4/5] Mobile API will be available at http://localhost:8082/api/mobile

REM -- Main Application (includes all services) ----------------------------
echo [5/5] Starting main application...
echo.
echo All services are embedded in the main application process:
echo   - Camera pipelines (detection, tracking, encoding)
echo   - MQTT event streaming
echo   - Context filter + VLM reasoning + orchestration
echo   - Alert system (push + webhook)
echo   - Watchdog health monitoring
echo   - Health API:   http://localhost:8080/api/health/devices
echo   - Dashboard:    http://localhost:8081/dashboard
echo   - Mobile API:   http://localhost:8082/api/mobile
echo.
echo Press Ctrl+C in the application window to stop.
echo.

pause
