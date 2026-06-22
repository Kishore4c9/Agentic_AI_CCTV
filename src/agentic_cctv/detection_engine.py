"""Detection Engine for the Agentic AI CCTV Monitoring Framework.

Runs the configured ``InferenceRuntime`` on each video frame and applies the
Detection Gate to filter results by confidence threshold and monitored object
classes.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

from agentic_cctv.models import CameraConfig, Detection, Frame, RawDetection
from agentic_cctv.runtimes import InferenceRuntime


def apply_detection_gate(detection: RawDetection, camera_config: CameraConfig) -> bool:
    """Return ``True`` iff the detection passes the Detection Gate.

    The gate passes when **both** conditions hold:

    * ``detection.confidence >= camera_config.confidence_threshold``
    * ``detection.object_type in camera_config.monitored_classes``
    """
    return (
        detection.confidence >= camera_config.confidence_threshold
        and detection.object_type in camera_config.monitored_classes
    )


class DetectionEngine:
    """Run inference on frames and apply the Detection Gate.

    Parameters
    ----------
    camera_config:
        Per-camera configuration containing the confidence threshold and
        monitored object classes used by the Detection Gate.
    runtime:
        An ``InferenceRuntime`` implementation (e.g. ``PyTorchRuntime``)
        used to run the detection model on each frame.
    """

    def __init__(self, camera_config: CameraConfig, runtime: InferenceRuntime) -> None:
        self._camera_config = camera_config
        self._runtime = runtime

    def detect(self, frame: Frame) -> list[Detection]:
        """Run inference on *frame* and return gated ``Detection`` results.

        Steps:

        1. Call ``self._runtime.infer(frame.image)`` to obtain raw detections.
        2. For each ``RawDetection``, evaluate the Detection Gate via
           :func:`apply_detection_gate`.
        3. Build a ``Detection`` for every raw detection, setting
           ``passed_gate`` accordingly.
        """
        raw_detections: list[RawDetection] = self._runtime.infer(frame.image)

        detections: list[Detection] = []
        for raw in raw_detections:
            passed = apply_detection_gate(raw, self._camera_config)
            detections.append(
                Detection(
                    object_type=raw.object_type,
                    bounding_box=raw.bounding_box,
                    confidence=raw.confidence,
                    passed_gate=passed,
                )
            )

        return detections
