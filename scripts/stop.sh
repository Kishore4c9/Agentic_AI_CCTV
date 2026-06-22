#!/usr/bin/env bash
# ==========================================================================
# Agentic AI CCTV Monitoring Framework — Stop All Services (Linux)
# ==========================================================================
# Stops all services started by launch.sh using saved PID files.
#
# Usage:
#   ./scripts/stop.sh
# ==========================================================================

PID_DIR="pids"

echo "============================================================"
echo " Agentic AI CCTV Monitoring Framework — Stopping Services"
echo "============================================================"
echo ""

_stop_service() {
    local name="$1"
    local pid_file="$PID_DIR/$name.pid"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (PID: $pid)..."
            kill "$pid"
            # Wait up to 10 seconds for graceful shutdown
            for i in $(seq 1 10); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing $name (PID: $pid)..."
                kill -9 "$pid" 2>/dev/null
            fi
            echo "  $name stopped."
        else
            echo "$name is not running (stale PID file)."
        fi
        rm -f "$pid_file"
    else
        echo "$name: no PID file found (not started by launch.sh?)."
    fi
}

_stop_service "application"

# Optionally stop Mosquitto if we started it
if [ -f "$PID_DIR/mosquitto.pid" ]; then
    _stop_service "mosquitto"
fi

echo ""
echo "All services stopped."
