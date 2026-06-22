"""Inference runtime abstractions for the Agentic AI CCTV Monitoring Framework.

Defines the ``InferenceRuntime`` protocol and concrete implementations:

* **PyTorchRuntime** – loads a YOLO v8 model via the Ultralytics library and
  runs inference on BGR numpy frames, returning a list of ``RawDetection``.
* **TensorRTRuntime** – placeholder stub for v1; raises ``NotImplementedError``
  so that the detection pipeline code remains runtime-agnostic while TensorRT
  optimisation is deferred to a later release.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from ultralytics import YOLO

from agentic_cctv.models import BoundingBox, RawDetection


@runtime_checkable
class InferenceRuntime(Protocol):
    """Protocol that all inference backends must satisfy."""

    def load_model(self, model_path: str) -> None:
        """Load a detection model from *model_path*."""
        ...

    def infer(self, image: np.ndarray) -> list[RawDetection]:
        """Run inference on a BGR *image* and return raw detections."""
        ...


class PyTorchRuntime:
    """YOLO v8 inference via the Ultralytics PyTorch backend."""

    def __init__(self) -> None:
        self._model: YOLO | None = None

    def load_model(self, model_path: str) -> None:
        """Load a YOLO v8 model from *model_path* using Ultralytics."""
        self._model = YOLO(model_path)

    def infer(self, image: np.ndarray) -> list[RawDetection]:
        """Run the loaded YOLO model on *image* and return ``RawDetection`` list.

        Each Ultralytics result box is mapped to a ``RawDetection`` with:
        * ``object_type`` – class name looked up via the model's ``names`` dict
        * ``bounding_box`` – ``BoundingBox(x, y, width, height)`` in pixel coords
        * ``confidence`` – detection confidence score
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self._model(image, verbose=False)

        detections: list[RawDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # xyxy format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                object_type = result.names.get(cls_id, str(cls_id))

                bounding_box = BoundingBox(
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1),
                )
                detections.append(
                    RawDetection(
                        object_type=object_type,
                        bounding_box=bounding_box,
                        confidence=conf,
                    )
                )

        return detections


class TensorRTRuntime:
    """Stub TensorRT runtime — not available in v1."""

    _MESSAGE = "TensorRT runtime is not available in v1. Use PyTorch runtime."

    def load_model(self, model_path: str) -> None:  # noqa: ARG002
        """Raise ``NotImplementedError``; TensorRT is deferred to a later release."""
        raise NotImplementedError(self._MESSAGE)

    def infer(self, image: np.ndarray) -> list[RawDetection]:  # noqa: ARG002
        """Raise ``NotImplementedError``; TensorRT is deferred to a later release."""
        raise NotImplementedError(self._MESSAGE)
