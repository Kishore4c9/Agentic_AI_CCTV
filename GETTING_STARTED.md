# Getting Started

This guide walks you through setting up, configuring, and running the Agentic AI CCTV Monitoring Framework on a Windows PC with an NVIDIA GPU.

## Prerequisites

Before you begin, make sure you have:

1. **Python 3.9 or later** — [Download from python.org](https://www.python.org/downloads/)
2. **NVIDIA GPU with CUDA** — RTX 3060 or above recommended
3. **NVIDIA CUDA Toolkit** — [Download from NVIDIA](https://developer.nvidia.com/cuda-downloads)
4. **Mosquitto MQTT Broker** — [Download from mosquitto.org](https://mosquitto.org/download/)
5. **NVIDIA Cosmos API Key** — [Get one from NVIDIA NIM](https://build.nvidia.com/)

## Step 1: Install the Project

```bash
# Clone the repository
git clone <repository-url>
cd video-agent

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS

# Install the project with development dependencies
pip install -e ".[dev]"
```

This installs all runtime dependencies (paho-mqtt, ultralytics, opencv-python, pyyaml, psutil, cryptography, aiohttp) and dev dependencies (pytest, hypothesis, pytest-asyncio, chromadb).

## Step 2: Download a YOLO Model

The detection engine uses YOLO v8. Download a pre-trained model:

```bash
# Create the models directory
mkdir models

# Download YOLOv8 nano (fastest, good for testing)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export()"
# The model file will be saved — move it to the models directory
move yolov8n.pt models\yolov8n.pt
```

Alternatively, download directly from [Ultralytics releases](https://github.com/ultralytics/assets/releases) and place the `.pt` file in the `models/` directory.

## Step 3: Start the MQTT Broker

Open a separate terminal and start Mosquitto:

```bash
mosquitto
```

By default, Mosquitto listens on `localhost:1883`. If you installed Mosquitto as a Windows service, it may already be running.

To verify the broker is running:

```bash
mosquitto_pub -t test -m "hello"
```

## Step 4: Create Your Configuration

```bash
# Copy the example config
copy config.example.yaml config.yaml
```

Open `config.yaml` in your editor and make these changes:

### Set your VLM API key

```yaml
vlm:
  backend: cosmos
  api_key: "nvapi-YOUR_REAL_API_KEY_HERE"   # Replace this
  endpoint: "https://integrate.api.nvidia.com/v1"
  timeout_seconds: 30
```

### Configure your cameras

Update the camera entries to match your setup:

```yaml
cameras:
  # USB webcam (use index "0" for the first webcam)
  - camera_id: cam-webcam
    uri: "0"
    tenant_id: my-tenant
    site_id: my-site
    confidence_threshold: 0.7
    monitored_classes:
      - person
      - vehicle
    inference_runtime: pytorch
    model_path: "./models/yolov8n.pt"
    tracker_algorithm: deepsort
    frame_skip: 3

  # RTSP IP camera
  - camera_id: cam-front-door
    uri: "rtsp://192.168.1.100:554/stream1"
    tenant_id: my-tenant
    site_id: my-site
    confidence_threshold: 0.6
    monitored_classes:
      - person
    inference_runtime: pytorch
    model_path: "./models/yolov8n.pt"
    tracker_algorithm: deepsort
    frame_skip: 5

  # Video file (for testing)
  - camera_id: cam-test
    uri: "./data/test_video.mp4"
    tenant_id: my-tenant
    site_id: my-site
    confidence_threshold: 0.5
    monitored_classes:
      - person
      - vehicle
    inference_runtime: pytorch
    model_path: "./models/yolov8n.pt"
    tracker_algorithm: deepsort
    frame_skip: 1
```

### Enable video snippet mode (optional)

To send video clips instead of single images to the VLM for better scene understanding:

```yaml
cameras:
  - camera_id: cam-parking
    uri: "rtsp://192.168.1.101:554/stream1"
    # ... other fields ...
    vlm_input_mode: video              # "image" (default) or "video"
    vlm_video_duration_seconds: 10     # Snippet duration: 1-60 seconds
```

When `vlm_input_mode` is `video`, the system buffers recent frames and assembles a short MP4 clip centred on each detection event. The default is `image` (single frame crop), which uses less memory and bandwidth.

### Set up encrypted frame crop storage (optional)

Generate an AES-256 encryption key and add it to the config:

## Step 4b: Simulate Cameras from Video Files (Optional)

If you don't have real IP cameras, you can stream local video files as camera feeds using the included simulator:

```bash
# Stream a single video (loops by default)
python scripts/rtsp_simulator.py path/to/lobby.mp4

# Stream multiple videos on separate endpoints
python scripts/rtsp_simulator.py lobby.mp4 parking.mp4 entrance.mp4

# Custom port and FPS override
python scripts/rtsp_simulator.py --port 9000 --fps 15 video.mp4

# Play once without looping
python scripts/rtsp_simulator.py --no-loop video.mp4
```

Each video gets its own endpoint — `stream1`, `stream2`, etc. Then point your cameras at them in `config.yaml`:

```yaml
cameras:
  - camera_id: cam-lobby
    uri: "http://localhost:8554/stream1"
    tenant_id: my-tenant
    site_id: my-site
    confidence_threshold: 0.7
    monitored_classes: [person, vehicle]
    inference_runtime: pytorch
    model_path: "./models/yolov8n.pt"
    tracker_algorithm: deepsort
    frame_skip: 3

  - camera_id: cam-parking
    uri: "http://localhost:8554/stream2"
    tenant_id: my-tenant
    site_id: my-site
    confidence_threshold: 0.6
    monitored_classes: [person]
    inference_runtime: pytorch
    model_path: "./models/yolov8n.pt"
    tracker_algorithm: deepsort
    frame_skip: 5
```

The simulator uses MJPEG-over-HTTP, which OpenCV handles natively — no extra dependencies needed. Videos loop continuously to mimic a live camera feed.

For true RTSP streams (`rtsp://` URIs), you can use FFmpeg + MediaMTX instead:

```bash
# 1. Download MediaMTX from https://github.com/bluenviron/mediamtx/releases
# 2. Start it:
./mediamtx

# 3. Push video via FFmpeg (loops forever):
ffmpeg -re -stream_loop -1 -i lobby.mp4 -c copy -f rtsp rtsp://localhost:8554/stream1
```

## Step 4c: Set Up Encrypted Frame Crop Storage (Optional)

Generate an AES-256 encryption key and add it to the config:

```bash
python -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"
```

```yaml
storage:
  frame_crop_encryption_key: "YOUR_BASE64_KEY_HERE"
```

## Step 5: Run the Application

You have two options: launch scripts (recommended) or manual launch.

### Option A: Launch Scripts (Recommended)

The `scripts/` directory contains ready-made launchers for both Windows and Linux.

**Windows:**

```cmd
scripts\launch.bat
```

This opens a new window for Mosquitto and starts the main application. All services (camera pipelines, VLM reasoning, dashboard, alerts) run together.

**Linux/Ubuntu:**

```bash
chmod +x scripts/*.sh
./scripts/launch.sh
```

This starts Mosquitto as a daemon and runs the application in the background. Logs go to `logs/application.log`.

To stop on Linux:

```bash
./scripts/stop.sh
```

### Option B: Development Mode

For development, use `launch-modules` — it adds an MQTT event monitor and test runner alongside the main app:

```cmd
REM Windows — opens separate terminal windows
scripts\launch-modules.bat
```

```bash
# Linux — starts background processes with individual log files
./scripts/launch-modules.sh

# Stop everything
./scripts/stop-modules.sh
```

### Option C: Manual Launch

If you prefer to start things yourself:

```bash
# Terminal 1: Start Mosquitto
mosquitto

# Terminal 2: Start the application
python -m agentic_cctv.main
# Or with a custom config path:
python -m agentic_cctv.main /path/to/config.yaml
```

You should see output like:

```
2025-01-15 14:30:00 [INFO] agentic_cctv.main: Starting Agentic AI CCTV Monitoring Framework…
2025-01-15 14:30:00 [INFO] agentic_cctv.main: MQTT publisher connected to localhost:1883 (TLS disabled).
2025-01-15 14:30:00 [INFO] agentic_cctv.main: TimeSeriesDB opened at './data/events.db'.
2025-01-15 14:30:00 [INFO] agentic_cctv.main: AlertSystem initialised with channels: ['push', 'webhook'].
2025-01-15 14:30:01 [INFO] agentic_cctv.main: Starting pipeline for camera 'cam-webcam' (uri=0).
2025-01-15 14:30:01 [INFO] agentic_cctv.main: All 1 camera pipeline(s) started. Entering main loop.
```

Press `Ctrl+C` to stop the application gracefully.

## Step 6: Access the Dashboard

Once running, the following endpoints are available:

| Service | URL | Description |
|---|---|---|
| Health API | http://localhost:8080/api/health/devices | Camera health status (JSON) |
| Dashboard | http://localhost:8081/dashboard | Operator web UI |
| Mobile API | http://localhost:8082/api/mobile | Mobile app REST endpoints |

## Using Environment Templates

Instead of configuring everything manually, you can use a pre-built template:

```yaml
# In config.yaml, just specify the template name:
environment_template: home

# Override specific values as needed:
vlm:
  api_key: "nvapi-YOUR_KEY"
cameras:
  - camera_id: cam-front
    uri: "0"
    tenant_id: my-home
    site_id: home
```

Available templates: `home`, `farm`, `forest`, `mall`, `port`, `gpu_desktop`

## Running Tests

```bash
# Run the full test suite (948 tests)
python -m pytest tests/ -q

# Run only property-based tests
python -m pytest tests/test_property_*.py -v

# Run tests for a specific component
python -m pytest tests/test_vlm_reasoner.py -v
python -m pytest tests/test_snippet_assembler.py -v
python -m pytest tests/test_config_manager.py -v
```

## Configuring Natural Language Rules

You can configure monitoring rules using plain English via the dashboard or mobile API:

1. Open the dashboard at http://localhost:8081/dashboard
2. Navigate to the Rules section
3. Enter a prompt like: "Alert me when a person is detected near the loading dock after 10pm"
4. Review the compiled rule set and confirm

The system compiles your prompt into a structured rule set and applies it to the specified camera(s) within 30 seconds.

## Troubleshooting

### MQTT broker not available

```
MQTT broker not available at localhost:1883 — continuing without MQTT.
```

Make sure Mosquitto is running. On Windows, check if the Mosquitto service is started, or run `mosquitto` in a separate terminal.

### Model not found

```
Failed to load model for camera 'cam-01'.
```

Verify the `model_path` in your config points to a valid `.pt` file in the `models/` directory.

### Camera connection failed

```
Failed to open video source 'rtsp://...' for camera 'cam-01'.
```

Check that the camera URI is correct and the camera is accessible from your network. For USB cameras, use the index as a string (e.g., `"0"`).

### VLM API errors

```
Cosmos API call failed: HTTP Error 401
```

Verify your API key is correct in `config.yaml`. Make sure you're using a valid NVIDIA Cosmos API key from [build.nvidia.com](https://build.nvidia.com/).

### ChromaDB not installed

```
chromadb not installed — VectorDB disabled.
```

Install the optional dependency: `pip install chromadb==1.0.7`

## Configuration Reference

For the complete configuration reference with all available fields and their defaults, see `config.example.yaml`. The file is fully documented with inline comments explaining each option.

Key configuration fields:

| Field | Default | Description |
|---|---|---|
| `deployment_profile` | `single-machine` | Deployment mode |
| `mqtt.host` | `localhost` | MQTT broker address |
| `mqtt.port` | `1883` | MQTT broker port |
| `vlm.backend` | `cosmos` | VLM provider |
| `vlm.api_key` | (required) | VLM API key |
| `cameras[].confidence_threshold` | `0.7` | Min detection confidence |
| `cameras[].monitored_classes` | `["person"]` | Object classes to detect |
| `cameras[].frame_skip` | `3` | Process every Nth frame |
| `cameras[].vlm_input_mode` | `image` | `image` or `video` |
| `cameras[].vlm_video_duration_seconds` | `10` | Video snippet duration (1-60s) |
| `storage.retention.raw_events_days` | `90` | Raw event retention |
| `alerts.cooldown.default_seconds` | `60` | Alert deduplication window |
