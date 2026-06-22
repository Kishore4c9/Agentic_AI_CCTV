"""Tests for the InferenceRuntime protocol, PyTorchRuntime, and TensorRTRuntime."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentic_cctv.models import BoundingBox, RawDetection
from agentic_cctv.runtimes import (
    InferenceRuntime,
    PyTorchRuntime,
    TensorRTRuntime,
)


# ---------------------------------------------------------------------------
# TensorRTRuntime — stub raises NotImplementedError
# ---------------------------------------------------------------------------


class TestTensorRTRuntime:
    """TensorRTRuntime should raise NotImplementedError for both methods."""

    def test_load_model_raises(self) -> None:
        rt = TensorRTRuntime()
        with pytest.raises(NotImplementedError, match="TensorRT runtime is not available in v1"):
            rt.load_model("any_path.pt")

    def test_infer_raises(self) -> None:
        rt = TensorRTRuntime()
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(NotImplementedError, match="TensorRT runtime is not available in v1"):
            rt.infer(dummy_image)


# ---------------------------------------------------------------------------
# PyTorchRuntime — model loading and inference
# ---------------------------------------------------------------------------


class TestPyTorchRuntime:
    """PyTorchRuntime wraps Ultralytics YOLO for inference."""

    def test_infer_without_load_raises(self) -> None:
        """Calling infer before load_model should raise RuntimeError."""
        rt = PyTorchRuntime()
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            rt.infer(dummy_image)

    @patch("agentic_cctv.runtimes.YOLO")
    def test_load_model_creates_yolo_instance(self, mock_yolo_cls: MagicMock) -> None:
        """load_model should instantiate YOLO with the given path."""
        rt = PyTorchRuntime()
        rt.load_model("models/yolov8n.pt")
        mock_yolo_cls.assert_called_once_with("models/yolov8n.pt")

    @patch("agentic_cctv.runtimes.YOLO")
    def test_infer_returns_raw_detections(self, mock_yolo_cls: MagicMock) -> None:
        """infer should map Ultralytics result boxes to RawDetection list."""
        # Build a fake Ultralytics result with two detections
        import torch

        box1 = MagicMock()
        box1.xyxy = torch.tensor([[100.0, 50.0, 300.0, 250.0]])
        box1.cls = torch.tensor([0])
        box1.conf = torch.tensor([0.92])

        box2 = MagicMock()
        box2.xyxy = torch.tensor([[400.0, 100.0, 500.0, 400.0]])
        box2.cls = torch.tensor([2])
        box2.conf = torch.tensor([0.75])

        result = MagicMock()
        result.boxes = [box1, box2]
        result.names = {0: "person", 1: "bicycle", 2: "car"}

        mock_model = MagicMock()
        mock_model.return_value = [result]
        mock_yolo_cls.return_value = mock_model

        rt = PyTorchRuntime()
        rt.load_model("models/yolov8n.pt")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = rt.infer(dummy_image)

        assert len(detections) == 2

        d0 = detections[0]
        assert isinstance(d0, RawDetection)
        assert d0.object_type == "person"
        assert d0.confidence == pytest.approx(0.92, abs=1e-4)
        assert d0.bounding_box == BoundingBox(x=100, y=50, width=200, height=200)

        d1 = detections[1]
        assert d1.object_type == "car"
        assert d1.confidence == pytest.approx(0.75, abs=1e-4)
        assert d1.bounding_box == BoundingBox(x=400, y=100, width=100, height=300)

    @patch("agentic_cctv.runtimes.YOLO")
    def test_infer_empty_results(self, mock_yolo_cls: MagicMock) -> None:
        """infer should return an empty list when no detections are found."""
        result = MagicMock()
        result.boxes = None

        mock_model = MagicMock()
        mock_model.return_value = [result]
        mock_yolo_cls.return_value = mock_model

        rt = PyTorchRuntime()
        rt.load_model("models/yolov8n.pt")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = rt.infer(dummy_image)

        assert detections == []

    @patch("agentic_cctv.runtimes.YOLO")
    def test_infer_unknown_class_id_uses_string_fallback(self, mock_yolo_cls: MagicMock) -> None:
        """When a class ID is not in the names dict, use its string representation."""
        import torch

        box = MagicMock()
        box.xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        box.cls = torch.tensor([999])
        box.conf = torch.tensor([0.5])

        result = MagicMock()
        result.boxes = [box]
        result.names = {0: "person"}  # 999 not present

        mock_model = MagicMock()
        mock_model.return_value = [result]
        mock_yolo_cls.return_value = mock_model

        rt = PyTorchRuntime()
        rt.load_model("models/yolov8n.pt")
        detections = rt.infer(np.zeros((100, 100, 3), dtype=np.uint8))

        assert len(detections) == 1
        assert detections[0].object_type == "999"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Both runtimes should satisfy the InferenceRuntime protocol."""

    def test_pytorch_runtime_is_inference_runtime(self) -> None:
        assert isinstance(PyTorchRuntime(), InferenceRuntime)

    def test_tensorrt_runtime_is_inference_runtime(self) -> None:
        assert isinstance(TensorRTRuntime(), InferenceRuntime)
