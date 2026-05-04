"""
Multi-model safety analyzer for construction site images.

Supports:
- YOLO11 (local ONNX) — fast, free, offline
- Claude Vision (Anthropic API) — VLM, zero-shot
- GPT-4o Vision (OpenAI API) — VLM, zero-shot
- Gemini 1.5 Pro (Google API) — VLM, zero-shot
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
VIOLATION_LABELS = [
    "no_helmet",
    "no_vest",
    "no_gloves",
    "no_boots",
    "unsafe_behavior",
    "restricted_area",
]

PPE_LABELS = [
    "helmet",
    "vest",
    "gloves",
    "boots",
    "person",
]

ALL_LABELS = VIOLATION_LABELS + PPE_LABELS


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[float] | None = None  # [x1, y1, x2, y2] or polygon points
    is_violation: bool = field(init=False)

    def __post_init__(self):
        self.is_violation = self.label in VIOLATION_LABELS


@dataclass
class AnalysisResult:
    model_name: str
    image_path: str
    detections: list[Detection]
    inference_time_ms: float
    raw_response: str | None = None  # for VLM models
    error: str | None = None

    @property
    def violations(self) -> list[Detection]:
        return [d for d in self.detections if d.is_violation]

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    @property
    def violation_labels(self) -> list[str]:
        return [d.label for d in self.violations]

    def summary(self) -> str:
        if self.error:
            return f"[{self.model_name}] ERROR: {self.error}"
        v = len(self.violations)
        t = len(self.detections)
        return f"[{self.model_name}] {t} detections, {v} violation(s) in {self.inference_time_ms:.1f}ms" + (
            f" → {self.violation_labels}" if v else ""
        )


# ---------------------------------------------------------------------------
# Base analyzer
# ---------------------------------------------------------------------------
class BaseAnalyzer(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def analyze(self, image_path: str | Path) -> AnalysisResult:
        image_path = str(image_path)
        start = time.perf_counter()
        try:
            detections = self._run_inference(image_path)
            error = None
        except Exception as e:
            detections = []
            error = str(e)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalysisResult(
            model_name=self.model_name,
            image_path=image_path,
            detections=detections,
            inference_time_ms=elapsed_ms,
            error=error,
        )

    @abstractmethod
    def _run_inference(self, image_path: str) -> list[Detection]: ...

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _get_media_type(image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}.get(
            ext.lstrip("."), "image/jpeg"
        )


# ---------------------------------------------------------------------------
# YOLO11 ONNX analyzer
# ---------------------------------------------------------------------------
class YOLOAnalyzer(BaseAnalyzer):
    """
    Uses your existing YOLO ONNX wrapper for local inference.
    Requires a trained ONNX model exported from training.py.
    Class names must match the Roboflow dataset classes.
    """

    # Map from dataset class index → label name
    # Update this to match your actual data.yaml class order
    DEFAULT_CLASS_NAMES = [
        "helmet",
        "no_helmet",
        "vest",
        "no_vest",
        "person",
        "gloves",
        "boots",
    ]

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
        self._model = None  # lazy load

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

    def _run_inference(self, image_path: str) -> list[Detection]:
        self._load_model()
        img = cv.imread(image_path)
        if img is None:
            msg = f"Cannot read image: {image_path}"
            raise ValueError(msg)

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


_SAFETY_SYSTEM_PROMPT = """You are a construction site safety inspector AI.
Analyze the image and detect PPE (Personal Protective Equipment) violations and unsafe behaviors.

For each person visible, check for:
- Helmet/hard hat (missing = no_helmet violation)
- High-visibility vest (missing = no_vest violation)
- Safety gloves (missing = no_gloves violation)
- Safety boots (missing = no_boots violation)
- Any unsafe posture, action, or presence in restricted areas (= unsafe_behavior)

Respond ONLY with a JSON array. Each element has:
{
  "label": "<one of: no_helmet, no_vest, no_gloves, no_boots, unsafe_behavior, helmet, vest, gloves, boots, person>",
  "confidence": <float 0.0-1.0>,
  "description": "<brief description>"
}

If no people or violations are found, return an empty array [].
Return ONLY the JSON, no markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# Azure AI Vision Analyzer (Foundry Endpoint)
# ---------------------------------------------------------------------------
class AzureAIVisionAnalyzer(BaseAnalyzer):
    """
    Uses Azure AI Vision 4.0 on Foundry.
    Endpoint: https://foundry-multimodal.cognitiveservices.azure.com/
    """

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        super().__init__("Azure-AI-Vision")
        self.api_key = api_key or os.getenv("AZURE_VISION_KEY", "")
        self.endpoint = endpoint or "https://foundry-multimodal.cognitiveservices.azure.com/"

    def _run_inference(self, image_path: str) -> list[Detection]:
        if not self.api_key:
            msg = "AZURE_VISION_KEY not set"
            raise RuntimeError(msg)

        with open(image_path, "rb") as f:
            image_data = f.read()

        params = urllib.parse.urlencode({"features": "people,objects", "api-version": "2023-02-01-preview"})
        url = f"{self.endpoint.rstrip('/')}/computervision/imageanalysis:analyze?{params}"

        req = urllib.request.Request(
            url,
            data=image_data,
            headers={
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": self.api_key,
            },
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        detections = []
        # Objects detected (helmets, vests, tools, etc.)
        for obj in data.get("objectsResult", {}).get("values", []):
            for tag in obj.get("tags", []):
                name = tag["name"].lower().replace(" ", "_")
                # Map Azure object names to our label schema
                label = _azure_label_map(name)
                bb = obj.get("boundingBox", {})
                bbox = None
                if bb:
                    x, y, w, h = bb.get("x", 0), bb.get("y", 0), bb.get("w", 0), bb.get("h", 0)
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                detections.append(Detection(label=label, confidence=tag["confidence"], bbox=bbox))

        # People detected — flag as "person" if no PPE context
        for person in data.get("peopleResult", {}).get("values", []):
            bb = person.get("boundingBox", {})
            bbox = None
            if bb:
                x, y, w, h = bb.get("x", 0), bb.get("y", 0), bb.get("w", 0), bb.get("h", 0)
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            detections.append(Detection(label="person", confidence=person.get("confidence", 0.9), bbox=bbox))

        return detections


# ---------------------------------------------------------------------------
# Foundry GPT-4o Analyzer
# ---------------------------------------------------------------------------
class FoundryGPT4oAnalyzer(BaseAnalyzer):
    """
    Uses GPT-4o deployed on Foundry.
    Endpoint: https://foundry-multimodal.cognitiveservices.azure.com/openai/deployments/gpt-4o/...
    """

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        super().__init__("Foundry-GPT4o")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_KEY", "")
        # Usiamo l'URL completo che hai fornito
        self.url = (
            endpoint
            or "https://foundry-multimodal.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
        )

    def _run_inference(self, image_path: str) -> list[Detection]:
        if not self.api_key:
            msg = "AZURE_OPENAI_KEY not set"
            raise RuntimeError(msg)

        image_data = self._encode_image_base64(image_path)
        media_type = self._get_media_type(image_path)

        payload = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": _SAFETY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this construction site image for safety violations."},
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                        ],
                    },
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json", "api-key": self.api_key}
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            response = json.loads(resp.read())

        raw = response["choices"][0]["message"]["content"].strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        items = json.loads(raw)
        detections = []
        for item in items:
            detections.append(
                Detection(label=item.get("label", "unknown"), confidence=float(item.get("confidence", 0.5)), bbox=None)
            )
        return detections


# ---------------------------------------------------------------------------
# Multi-model runner
# ---------------------------------------------------------------------------
def _azure_label_map(azure_name: str) -> str:
    """Map Azure Computer Vision object names to our safety label schema."""
    _MAP = {
        "helmet": "helmet",
        "hard_hat": "helmet",
        "hardhat": "helmet",
        "safety_helmet": "helmet",
        "construction_helmet": "helmet",
        "vest": "vest",
        "safety_vest": "vest",
        "high_visibility_vest": "vest",
        "hi-vis_vest": "vest",
        "reflective_vest": "vest",
        "glove": "gloves",
        "gloves": "gloves",
        "safety_gloves": "gloves",
        "boot": "boots",
        "boots": "boots",
        "safety_boot": "boots",
        "person": "person",
        "man": "person",
        "woman": "person",
        "worker": "person",
        "human": "person",
        "construction_worker": "person",
    }
    return _MAP.get(azure_name, azure_name)


class SafetyAnalyzerPipeline:
    """
    Runs multiple analyzers on one or more images and collects results.
    """

    def __init__(self, analyzers: list[BaseAnalyzer]):
        self.analyzers = analyzers

    def run(self, image_path: str | Path) -> list[AnalysisResult]:
        return [a.analyze(image_path) for a in self.analyzers]

    def run_batch(self, image_paths: list[str | Path]) -> dict[str, list[AnalysisResult]]:
        return {str(p): self.run(p) for p in image_paths}

    @staticmethod
    def print_report(results: list[AnalysisResult]) -> None:
        print("\n" + "=" * 60)
        print("SAFETY ANALYSIS REPORT")
        print("=" * 60)
        for r in results:
            print(r.summary())
            if r.violations:
                for v in r.violations:
                    print(f"  ⚠️  {v.label} (conf={v.confidence:.2f})")
        print("=" * 60 + "\n")
