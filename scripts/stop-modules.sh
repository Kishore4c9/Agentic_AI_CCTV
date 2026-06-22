#!/usr/bin/env bash
# ==========================================================================
# Agentic AI CCTV Monitoring Framework — Stop All Modules (Linux)
# ==========================================================================
# Stops all services started by launch-modules.sh.
#
# Usage:
#   ./scripts/stop-modules.sh
# ==========================================================================

PID_DIR="pids"

echo "Stopping all modules..."

_stop() {
    local name="$1"
    local pid_file="$PID_DIR/$name.pid"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            sleep 1
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
            echo "  Stopped $name (PID: $pid)"
        fi
        rm -f "$pid_file"
    fi
}

_stop "application"
_stop "mqtt-monitor"
_stop "mosquitto"

echo "Done."
