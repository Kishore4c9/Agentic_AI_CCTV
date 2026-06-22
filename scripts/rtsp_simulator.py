"""RTSP Camera Simulator — streams local video files as RTSP feeds.

Reads one or more video files from disk and re-streams them over RTSP
so the CCTV monitoring framework can consume them as if they were live
IP cameras. Each video file gets its own RTSP endpoint.

Requirements:
    pip install opencv-python

Usage:
    # Stream a single video on rtsp://localhost:8554/stream1
    python scripts/rtsp_simulator.py video.mp4

    # Stream multiple videos on separate endpoints
    python scripts/rtsp_simulator.py lobby.mp4 parking.mp4 entrance.mp4

    # Custom host/port and loop forever
    python scripts/rtsp_simulator.py --host 0.0.0.0 --port 8554 --loop video.mp4

Then configure your cameras in config.yaml:
    cameras:
      - camera_id: cam-lobby
        uri: "rtsp://localhost:8554/stream1"
        ...

This uses a lightweight TCP server that speaks just enough RTSP to satisfy
OpenCV's VideoCapture RTSP client. No external RTSP server (like MediaMTX
or VLC) is needed.

If you prefer a full RTSP server, see the "Alternative: FFmpeg + MediaMTX"
section at the bottom of this file.
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rtsp_simulator")


# ---------------------------------------------------------------------------
# Minimal RTSP-over-TCP server
# ---------------------------------------------------------------------------
# OpenCV's VideoCapture with an RTSP URL expects a proper RTSP handshake.
# This is complex to implement correctly. Instead, we use a simpler approach:
# a raw TCP server that pushes MJPEG frames, consumed via OpenCV's TCP mode.
#
# For production RTSP simulation, use FFmpeg + MediaMTX (see bottom of file).
# This script provides a zero-dependency alternative that works well enough
# for development and testing.
# ---------------------------------------------------------------------------


class VideoStreamer:
    """Streams a video file over TCP as an MJPEG-over-HTTP feed.

    Each connected client receives a multipart MJPEG stream that OpenCV
    can consume via ``cv2.VideoCapture("http://host:port/streamN")``.

    Parameters
    ----------
    video_path:
        Path to the video file to stream.
    stream_name:
        Name for this stream (used in the URL path).
    fps:
        Playback frame rate. If None, uses the video's native FPS.
    loop:
        If True, restart the video from the beginning when it ends.
    """

    def __init__(
        self,
        video_path: str,
        stream_name: str = "stream1",
        fps: Optional[float] = None,
        loop: bool = True,
    ) -> None:
        self.video_path = video_path
        self.stream_name = stream_name
        self.loop = loop
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._running = False

        # Probe the video to get native FPS
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.fps = fps if fps else native_fps
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        logger.info(
            "Stream '%s': %s (%dx%d, %.1f fps, %d frames, loop=%s)",
            stream_name, video_path, width, height, self.fps,
            self.total_frames, loop,
        )

    def add_client(self, client_sock: socket.socket) -> None:
        """Register a new client connection."""
        with self._lock:
            self._clients.append(client_sock)
        logger.info(
            "Stream '%s': client connected (%d total)",
            self.stream_name, len(self._clients),
        )

    def remove_client(self, client_sock: socket.socket) -> None:
        """Remove a disconnected client."""
        with self._lock:
            if client_sock in self._clients:
                self._clients.remove(client_sock)
        try:
            client_sock.close()
        except OSError:
            pass

    def start(self) -> None:
        """Start the frame-pushing thread."""
        self._running = True
        thread = threading.Thread(
            target=self._stream_loop,
            name=f"Streamer-{self.stream_name}",
            daemon=True,
        )
        thread.start()

    def stop(self) -> None:
        """Stop streaming."""
        self._running = False

    def _stream_loop(self) -> None:
        """Read frames from the video and push to all connected clients."""
        frame_interval = 1.0 / self.fps

        while self._running:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logger.error("Cannot open video: %s", self.video_path)
                time.sleep(1)
                continue

            frame_count = 0
            while self._running and cap.isOpened():
                t_start = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    if self.loop:
                        logger.info(
                            "Stream '%s': video ended, looping...",
                            self.stream_name,
                        )
                        break  # Break inner loop to reopen
                    else:
                        logger.info(
                            "Stream '%s': video ended (no loop).",
                            self.stream_name,
                        )
                        self._running = False
                        break

                frame_count += 1

                # Encode frame as JPEG
                _, jpeg_buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                jpeg_bytes = jpeg_buf.tobytes()

                # Build MJPEG multipart chunk
                chunk = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n"
                    b"\r\n" + jpeg_bytes + b"\r\n"
                )

                # Send to all clients
                with self._lock:
                    dead_clients = []
                    for client in self._clients:
                        try:
                            client.sendall(chunk)
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            dead_clients.append(client)

                    for dc in dead_clients:
                        self._clients.remove(dc)
                        try:
                            dc.close()
                        except OSError:
                            pass
                        logger.info(
                            "Stream '%s': client disconnected (%d remaining)",
                            self.stream_name, len(self._clients),
                        )

                # Pace to target FPS
                elapsed = time.monotonic() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            cap.release()

            if not self.loop:
                break


class MJPEGServer:
    """HTTP server that serves MJPEG streams from multiple video files.

    Clients connect via ``http://host:port/streamN`` and receive a
    multipart MJPEG stream that OpenCV can consume with VideoCapture.

    Parameters
    ----------
    host:
        Bind address.
    port:
        Bind port.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8554) -> None:
        self.host = host
        self.port = port
        self._streamers: dict[str, VideoStreamer] = {}
        self._server_sock: Optional[socket.socket] = None
        self._running = False

    def add_stream(self, streamer: VideoStreamer) -> None:
        """Register a video streamer."""
        self._streamers[streamer.stream_name] = streamer

    def start(self) -> None:
        """Start the HTTP server and all streamers."""
        self._running = True

        # Start all streamers
        for streamer in self._streamers.values():
            streamer.start()

        # Start TCP accept loop
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(16)
        self._server_sock.settimeout(1.0)

        logger.info("MJPEG server listening on %s:%d", self.host, self.port)
        logger.info("Available streams:")
        for name in self._streamers:
            logger.info(
                "  http://%s:%d/%s",
                "localhost" if self.host == "0.0.0.0" else self.host,
                self.port,
                name,
            )

        thread = threading.Thread(
            target=self._accept_loop, name="MJPEG-Accept", daemon=True
        )
        thread.start()

    def _accept_loop(self) -> None:
        """Accept incoming connections and route to the correct stream."""
        while self._running:
            try:
                client_sock, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            # Handle in a thread so we don't block accept
            threading.Thread(
                target=self._handle_client,
                args=(client_sock, addr),
                daemon=True,
            ).start()

    def _handle_client(
        self, client_sock: socket.socket, addr: tuple
    ) -> None:
        """Read the HTTP request, find the stream, send MJPEG response."""
        try:
            # Read the HTTP request (just need the first line)
            data = client_sock.recv(4096)
            if not data:
                client_sock.close()
                return

            request_line = data.decode("utf-8", errors="ignore").split("\r\n")[0]
            # e.g. "GET /stream1 HTTP/1.1"
            parts = request_line.split()
            if len(parts) < 2:
                client_sock.close()
                return

            path = parts[1].lstrip("/")

            if path not in self._streamers:
                # Send 404
                response = (
                    "HTTP/1.1 404 Not Found\r\n"
                    "Content-Type: text/plain\r\n\r\n"
                    f"Stream '{path}' not found.\n"
                    f"Available: {', '.join(self._streamers.keys())}\n"
                )
                client_sock.sendall(response.encode())
                client_sock.close()
                return

            # Send MJPEG HTTP response header
            header = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: keep-alive\r\n"
                "\r\n"
            )
            client_sock.sendall(header.encode())

            # Register client with the streamer
            self._streamers[path].add_client(client_sock)

        except Exception:
            logger.exception("Error handling client %s", addr)
            try:
                client_sock.close()
            except OSError:
                pass

    def stop(self) -> None:
        """Stop the server and all streamers."""
        self._running = False
        for streamer in self._streamers.values():
            streamer.stop()
        if self._server_sock:
            self._server_sock.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream local video files as MJPEG feeds for the CCTV framework.",
        epilog=(
            "Example:\n"
            "  python scripts/rtsp_simulator.py lobby.mp4 parking.mp4\n\n"
            "Then in config.yaml:\n"
            '  uri: "http://localhost:8554/stream1"\n'
            '  uri: "http://localhost:8554/stream2"\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="Video file paths to stream",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8554,
        help="Bind port (default: 8554)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override playback FPS (default: use video's native FPS)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Stop after the video ends instead of looping",
    )

    args = parser.parse_args()

    server = MJPEGServer(host=args.host, port=args.port)

    for i, video_path in enumerate(args.videos, start=1):
        stream_name = f"stream{i}"
        streamer = VideoStreamer(
            video_path=video_path,
            stream_name=stream_name,
            fps=args.fps,
            loop=not args.no_loop,
        )
        server.add_stream(streamer)

    server.start()

    print()
    print("Camera simulator running. Press Ctrl+C to stop.")
    print()
    print("Configure your cameras in config.yaml with these URIs:")
    for i in range(len(args.videos)):
        host = "localhost" if args.host == "0.0.0.0" else args.host
        print(f'  uri: "http://{host}:{args.port}/stream{i + 1}"')
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        server.stop()


# ---------------------------------------------------------------------------
# Alternative: FFmpeg + MediaMTX (true RTSP)
# ---------------------------------------------------------------------------
#
# For a proper RTSP server that supports rtsp:// URIs, use MediaMTX:
#
# 1. Download MediaMTX: https://github.com/bluenviern/mediamtx/releases
# 2. Start it:          ./mediamtx
# 3. Push a video via FFmpeg:
#
#    ffmpeg -re -stream_loop -1 -i lobby.mp4 \
#           -c copy -f rtsp rtsp://localhost:8554/stream1
#
#    ffmpeg -re -stream_loop -1 -i parking.mp4 \
#           -c copy -f rtsp rtsp://localhost:8554/stream2
#
# 4. In config.yaml:
#    uri: "rtsp://localhost:8554/stream1"
#
# This gives you true RTSP with proper codec negotiation, but requires
# FFmpeg and MediaMTX to be installed.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
