"""
Safety analyzer for construction site PPE detection.

Core pipeline: YOLO (local ONNX) for fast object detection.
VLM escalation is handled separately via vlm_validator module.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2 as cv

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
VIOLATION_LABELS = ["Non-Helmet", "Bare-arm", "no_helmet", "no_vest", "unsafe_behavior"]

PPE_LABELS = ["Glove", "Helmet", "Person", "Shoe", "Vest"]

ALL_LABELS = VIOLATION_LABELS + PPE_LABELS


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[float] | None = None
    is_violation: bool = field(init=False)

    def __post_init__(self):
        self.is_violation = self.label in VIOLATION_LABELS


@dataclass
class AnalysisResult:
    model_name: str
    image_path: str
    detections: list[Detection]
    inference_time_ms: float
    error: str | None = None

    @property
    def violations(self) -> list[Detection]:
        return [d for d in self.detections if d.is_violation]

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def summary(self) -> str:
        if self.error:
            return f"[{self.model_name}] ERROR: {self.error}"

        v = len(self.violations)
        t = len(self.detections)
        det_list = ", ".join([f"{d.label}({d.confidence:.2f})" for d in self.detections])
        base_info = f"[{self.model_name}] {t} detections ({det_list}) in {self.inference_time_ms:.1f}ms"

        if v:
            return base_info + f" | {v} VIOLATION(S): {[d.label for d in self.violations]}"
        return base_info


# ---------------------------------------------------------------------------
# Base analyzer
# ---------------------------------------------------------------------------
class BaseAnalyzer(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def analyze(self, image_source: str | Path | np.ndarray) -> AnalysisResult:
        import numpy as _np

        label = "<ndarray>" if isinstance(image_source, _np.ndarray) else str(image_source)
        start = time.perf_counter()
        try:
            detections = self._run_inference(image_source)
            error = None
        except Exception as e:
            detections = []
            error = str(e)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalysisResult(
            model_name=self.model_name,
            image_path=label,
            detections=detections,
            inference_time_ms=elapsed_ms,
            error=error,
        )

    @abstractmethod
    def _run_inference(self, image_source: str | np.ndarray) -> list[Detection]: ...

    @staticmethod
    def _to_bgr(image_source: str | Path | np.ndarray) -> np.ndarray:
        import numpy as np

        if isinstance(image_source, np.ndarray):
            return image_source
        img = cv.imread(str(image_source))
        if img is None:
            msg = f"Impossibile leggere l'immagine: {image_source}"
            raise ValueError(msg)
        return img


# ---------------------------------------------------------------------------
# YOLO ONNX analyzer
# ---------------------------------------------------------------------------
class YOLOAnalyzer(BaseAnalyzer):
    """
    Local YOLO inference via ONNX backend.
    Class names must match the Roboflow dataset class order.
    """

    DEFAULT_CLASS_NAMES = ["Glove", "Helmet", "Non-Helmet", "Person", "Shoe", "Vest", "Bare-arm"]

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.4,
        imgsz: int = 640,
        class_names: list[str] | None = None,
    ):
        super().__init__("YOLO11-ONNX")
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            yolo_dir = Path(__file__).resolve().parent.parent.parent / "yolo" / "src"
            if str(yolo_dir) not in sys.path:
                sys.path.insert(0, str(yolo_dir))

            from yolo.yolo_onnx import YOLOParams, YOLOPredictor
            from yolo.yolo_onnx.backends.onnxruntime import ONNXRuntimeBackend, ONNXRuntimeParams

            backend = ONNXRuntimeBackend(ONNXRuntimeParams(self.model_path))
            self._model = YOLOPredictor(YOLOParams(self.imgsz, conf=self.conf_threshold), backend)
        except ImportError as e:
            msg = f"Cannot import YOLO ONNX backend: {e}. Make sure the yolo subpackage is on your PYTHONPATH."
            raise RuntimeError(msg)

    def _run_inference(self, image_source: str | np.ndarray) -> list[Detection]:
        self._load_model()
        img = self._to_bgr(image_source)
        results = self._model.predict(img)

        detections = []
        for r in results:
            if r.conf < self.conf_threshold:
                continue
            label = self.class_names[r.class_id] if r.class_id < len(self.class_names) else f"class_{r.class_id}"
            detections.append(
                Detection(
                    label=label,
                    confidence=r.conf,
                    bbox=r.bbox,
                )
            )
        return detections
