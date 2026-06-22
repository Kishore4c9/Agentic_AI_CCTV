#!/usr/bin/env bash
# ==========================================================================
# Agentic AI CCTV Monitoring Framework — Ubuntu/Linux Launcher
# ==========================================================================
# Starts all services: Mosquitto MQTT broker and the main application.
# Each service runs in the background with logs written to logs/.
#
# Usage:
#   ./scripts/launch.sh                  (uses config.yaml)
#   ./scripts/launch.sh myconfig.yaml    (uses custom config)
#
# To stop all services:
#   ./scripts/stop.sh
# ==========================================================================

set -e

CONFIG_FILE="${1:-config.yaml}"
LOG_DIR="logs"
PID_DIR="pids"

echo "============================================================"
echo " Agentic AI CCTV Monitoring Framework — Launcher (Linux)"
echo "============================================================"
echo ""
echo "Config file: $CONFIG_FILE"
echo ""

# -- Create directories ----------------------------------------------------
mkdir -p "$LOG_DIR" "$PID_DIR" data models

# -- Check Python is available ---------------------------------------------
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Install with: sudo apt install python3 python3-pip"
    exit 1
fi

PYTHON=$(command -v python3)
echo "Using Python: $($PYTHON --version)"

# -- Check config file exists ----------------------------------------------
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[WARN] Config file '$CONFIG_FILE' not found."
    if [ -f "config.example.yaml" ]; then
        cp config.example.yaml "$CONFIG_FILE"
        echo "       Created $CONFIG_FILE from config.example.yaml — edit it before running."
    else
        echo "       No config.example.yaml found. The app will generate defaults."
    fi
fi

# -- Start Mosquitto MQTT Broker -------------------------------------------
echo "[1/2] Starting Mosquitto MQTT broker..."
if command -v mosquitto &> /dev/null; then
    # Check if already running
    if pgrep -x mosquitto > /dev/null 2>&1; then
        echo "       Mosquitto is already running (PID: $(pgrep -x mosquitto))."
    else
        mosquitto -d -c /etc/mosquitto/mosquitto.conf 2>/dev/null \
            || mosquitto -d 2>/dev/null \
            || echo "[WARN] Failed to start Mosquitto. Start it manually: sudo systemctl start mosquitto"
        sleep 1
        if pgrep -x mosquitto > /dev/null 2>&1; then
            echo "       Mosquitto started (PID: $(pgrep -x mosquitto))."
            pgrep -x mosquitto > "$PID_DIR/mosquitto.pid"
        fi
    fi
else
    echo "[WARN] Mosquitto not installed. Install with: sudo apt install mosquitto"
    echo "       Or start it manually before running the application."
fi

# -- Start Main Application ------------------------------------------------
echo "[2/2] Starting main application..."
echo ""

nohup $PYTHON -m agentic_cctv.main "$CONFIG_FILE" \
    > "$LOG_DIR/application.log" 2>&1 &
APP_PID=$!
echo "$APP_PID" > "$PID_DIR/application.pid"

echo "Main application started (PID: $APP_PID)"
echo "Log file: $LOG_DIR/application.log"
echo ""
echo "Services available:"
echo "  - Health API:   http://localhost:8080/api/health/devices"
echo "  - Dashboard:    http://localhost:8081/dashboard"
echo "  - Mobile API:   http://localhost:8082/api/mobile"
echo ""
echo "To view logs:    tail -f $LOG_DIR/application.log"
echo "To stop:         ./scripts/stop.sh"
echo ""
