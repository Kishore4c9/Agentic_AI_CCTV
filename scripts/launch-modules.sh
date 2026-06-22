#!/usr/bin/env bash
# ==========================================================================
# Agentic AI CCTV Monitoring Framework — Module Launcher (Ubuntu/Linux)
# ==========================================================================
# Launches individual modules in separate background processes for
# development and debugging. Each module logs to its own file.
#
# Usage:
#   ./scripts/launch-modules.sh                  (uses config.yaml)
#   ./scripts/launch-modules.sh myconfig.yaml    (uses custom config)
#
# To stop all:
#   ./scripts/stop-modules.sh
# ==========================================================================

set -e

CONFIG_FILE="${1:-config.yaml}"
LOG_DIR="logs"
PID_DIR="pids"
PYTHON=$(command -v python3 || command -v python)

echo "============================================================"
echo " Module Launcher — Individual Services (Linux)"
echo "============================================================"
echo ""
echo "Config: $CONFIG_FILE"
echo "Logs:   $LOG_DIR/"
echo ""

mkdir -p "$LOG_DIR" "$PID_DIR" data models

# -- 1. Mosquitto MQTT Broker ----------------------------------------------
echo "[1/5] Mosquitto MQTT Broker"
if command -v mosquitto &> /dev/null; then
    if pgrep -x mosquitto > /dev/null 2>&1; then
        echo "       Already running (PID: $(pgrep -x mosquitto))"
    else
        mosquitto -d 2>/dev/null && sleep 1
        if pgrep -x mosquitto > /dev/null 2>&1; then
            pgrep -x mosquitto > "$PID_DIR/mosquitto.pid"
            echo "       Started (PID: $(cat $PID_DIR/mosquitto.pid))"
        else
            echo "       [WARN] Failed to start. Run manually: mosquitto -d"
        fi
    fi
else
    echo "       [WARN] Not installed. Install: sudo apt install mosquitto"
fi

# -- 2. MQTT Event Monitor -------------------------------------------------
echo "[2/5] MQTT Event Monitor"
if command -v mosquitto_sub &> /dev/null; then
    nohup mosquitto_sub -t "+/+/+/#" -v > "$LOG_DIR/mqtt-monitor.log" 2>&1 &
    echo "$!" > "$PID_DIR/mqtt-monitor.pid"
    echo "       Started (PID: $!) — Log: $LOG_DIR/mqtt-monitor.log"
else
    echo "       [SKIP] mosquitto_sub not found"
fi

# -- 3. Main Application (all pipelines + servers) -------------------------
echo "[3/5] Main Application (detection + VLM + alerts + dashboard)"
nohup $PYTHON -m agentic_cctv.main "$CONFIG_FILE" > "$LOG_DIR/application.log" 2>&1 &
echo "$!" > "$PID_DIR/application.pid"
echo "       Started (PID: $!) — Log: $LOG_DIR/application.log"

# -- 4. Wait for servers to start -----------------------------------------
echo "[4/5] Waiting for servers to initialize..."
sleep 3

# -- 5. Verify services ----------------------------------------------------
echo "[5/5] Verifying services..."
echo ""

_check_port() {
    local name="$1"
    local port="$2"
    local url="$3"
    if command -v curl &> /dev/null; then
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" 2>/dev/null | grep -q "200\|404"; then
            echo "  ✓ $name — http://localhost:$port$url"
        else
            echo "  ○ $name — http://localhost:$port$url (not ready yet)"
        fi
    else
        echo "  ? $name — http://localhost:$port$url (install curl to verify)"
    fi
}

_check_port "Health API" 8080 "/api/health/devices"
_check_port "Dashboard"  8081 "/dashboard"
_check_port "Mobile API" 8082 "/api/mobile"

echo ""
echo "============================================================"
echo " All modules launched."
echo ""
echo " View logs:   tail -f $LOG_DIR/application.log"
echo " Stop all:    ./scripts/stop-modules.sh"
echo "============================================================"
