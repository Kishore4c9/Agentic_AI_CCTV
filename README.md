# Agentic AI CCTV Monitoring Framework

An event-driven surveillance platform that combines edge AI object detection, MQTT-based event streaming, context-aware filtering, Vision-Language Model (VLM) reasoning, and agentic orchestration into a single Python codebase.

The system uses a two-gate pipeline to minimize VLM compute costs: a Detection Gate filters by confidence and object class at the edge, and a Context Gate evaluates events against per-camera rule sets. Only events passing both gates (~2-5% of frames) trigger VLM reasoning.

## Features

- **Edge Object Detection** — YOLO v8 via PyTorch with configurable confidence thresholds and monitored object classes
- **Multi-Object Tracking** — DeepSORT or ByteTrack with persistent track IDs across frames (30-frame occlusion tolerance)
- **MQTT Event Streaming** — Structured events published via MQTT with QoS 1/2, store-and-forward for zero event loss during outages
- **Context Filtering** — Per-camera rule sets with object type, confidence, time window, zone (point-in-polygon), and compound conditions
- **VLM Reasoning** — NVIDIA Cosmos (default), with pluggable GPT-4o, Claude 3, and Gemini 1.5 backends. Supports image crops or configurable video snippets (1-60 seconds)
- **Agentic Orchestration** — LangChain-based agent deciding alert, log, summarise, or escalate actions with cross-camera context via MCP and A2A protocols
- **Natural Language Prompts** — Configure monitoring rules using plain English, compiled to structured rule sets via LLM
- **Alert System** — Push notifications and webhooks with configurable cooldown deduplication
- **Camera Health Monitoring** — Watchdog with 60-second offline detection, heartbeat metrics, and real-time health dashboard
- **Operator Dashboard** — Web UI for camera health, live event feed, alert management, and rule configuration
- **Mobile API** — REST endpoints for alert management, prompt configuration, and push notification delivery
- **Multi-Tenant Isolation** — All queries scoped by tenant ID, AES-256 encrypted frame crop storage
- **Environment Templates** — Pre-built configs for home, farm, forest, mall, port, and GPU desktop deployments

## Architecture

```
Camera → VideoFeeder → DetectionEngine → Tracker → EventEncoder → MQTT Broker
                                                                       ↓
                                              ContextFilter ← Rule Store
                                                    ↓ (pass)
                                              VLM Reasoner (Cosmos / GPT-4o / ...)
                                                    ↓
                                           Orchestration Agent (LangChain)
                                                    ↓
                                    Alert System → Push / Webhook / Dashboard
```

## Requirements

- Python 3.9+
- Windows 10/11 (primary target) or Linux
- NVIDIA GPU with CUDA support (for PyTorch inference)
- Mosquitto MQTT broker (local or remote)
- NVIDIA Cosmos API key (or other supported VLM provider)

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed setup instructions.

```bash
# 1. Clone and install
git clone <repository-url>
cd video-agent
pip install -e ".[dev]"

# 2. Configure
cp config.example.yaml config.yaml
# Edit config.yaml — set your VLM API key and camera URIs

# 3. Launch (Windows)
scripts\launch.bat

# 3. Launch (Linux/Ubuntu)
chmod +x scripts/*.sh
./scripts/launch.sh
```

This starts the Mosquitto MQTT broker and the main application. All services (detection pipelines, VLM reasoning, dashboard, mobile API) run in a single process.

For development, use `launch-modules` instead — it adds an MQTT event monitor and test runner in separate windows:

```bash
# Windows
scripts\launch-modules.bat

# Linux
./scripts/launch-modules.sh
```

## Launch Scripts

| Script | Platform | Purpose |
|---|---|---|
| `scripts/launch.bat` | Windows | Start Mosquitto + main app (production) |
| `scripts/launch.sh` | Linux | Start Mosquitto + main app as background daemon |
| `scripts/launch-modules.bat` | Windows | Start all services in separate windows (development) |
| `scripts/launch-modules.sh` | Linux | Start all services with individual log files (development) |
| `scripts/stop.sh` | Linux | Stop services started by `launch.sh` |
| `scripts/stop-modules.sh` | Linux | Stop services started by `launch-modules.sh` |

## Project Structure

```
src/agentic_cctv/
├── main.py                 # Application entry point
├── models.py               # Shared data models (dataclasses)
├── config_manager.py       # YAML config loading and validation
├── video_feeder.py         # Camera capture + FrameRingBuffer
├── detection_engine.py     # YOLO detection + Detection Gate
├── runtimes.py             # PyTorch / TensorRT inference backends
├── tracker.py              # DeepSORT / ByteTrack multi-object tracking
├── event_encoder.py        # StructuredEvent creation + video snippets
├── snippet_assembler.py    # MP4 video snippet assembly from buffered frames
├── mqtt_client.py          # MQTT publisher / subscriber (paho-mqtt)
├── store_and_forward.py    # SQLite-backed offline message queue
├── timeseries_db.py        # SQLite event / alert / heartbeat persistence
├── context_filter.py       # Rule-based event filtering (Context Gate)
├── rule_store.py           # Versioned rule set storage with rollback
├── vlm_reasoner.py         # VLM invocation with retry + fallback
├── vlm_backends.py         # Cosmos / GPT-4o / Claude 3 / Gemini 1.5
├── orchestration_agent.py  # LangChain agent + tool chain
├── alert_system.py         # Push / webhook delivery + cooldown
├── prompt_compiler.py      # Natural language → rule set compilation
├── watchdog.py             # Camera health monitoring
├── heartbeat_publisher.py  # Periodic health metric publishing
├── mcp_server.py           # Cross-camera context sharing (MCP)
├── a2a_comm.py             # Agent-to-agent communication (A2A)
├── frame_crop_store.py     # AES-256 encrypted crop storage
├── retention_scheduler.py  # Data retention enforcement
├── event_summarizer.py     # Hourly / daily natural language summaries
├── environment_templates.py# Pre-built deployment templates
├── dashboard.py            # Web UI operator dashboard
├── health_api.py           # REST API for device health
├── mobile_api.py           # Mobile REST API endpoints
├── phase2_pipeline.py      # Phase 2 MQTT subscriber wiring
└── vector_db.py            # ChromaDB vector storage
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -q

# Run with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_vlm_reasoner.py -v

# Run property-based tests only
python -m pytest tests/test_property_*.py -v
```

The test suite includes 948 tests: unit tests, property-based tests (Hypothesis), and integration tests across all four implementation phases.

## Configuration

All settings are in a single `config.yaml` file. See `config.example.yaml` for a fully documented example. Key sections:

| Section | Purpose |
|---|---|
| `deployment_profile` | `single-machine`, `multi-machine`, or `edge-cloud-hybrid` |
| `mqtt` | Broker host, port, TLS, authentication |
| `vlm` | Backend selection (`cosmos`, `gpt4o`, `claude3`, `gemini15`), API key |
| `cameras` | Per-camera URI, thresholds, tracked classes, VLM input mode |
| `storage` | Database paths, encryption key, retention policies |
| `alerts` | Delivery channels, webhook URL, cooldown settings |
| `security` | TLS, OAuth, tenant isolation |

## License

Proprietary
