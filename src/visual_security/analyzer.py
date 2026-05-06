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

    # def summary(self) -> str:
    #     if self.error:
    #         return f"[{self.model_name}] ERROR: {self.error}"
    #     v = len(self.violations)
    #     t = len(self.detections)
    #     return f"[{self.model_name}] {t} detections, {v} violation(s) in {self.inference_time_ms:.1f}ms" + (
    #         f" → {self.violation_labels}" if v else ""
    #     )

    def summary(self) -> str:
        if self.error:
            return f"[{self.model_name}] ERROR: {self.error}"

        v = len(self.violations)
        t = len(self.detections)

        det_list = ", ".join([f"{d.label}({d.confidence:.2f})" for d in self.detections])

        base_info = f"[{self.model_name}] {t} detections ({det_list}) in {self.inference_time_ms:.1f}ms"

        if v:
            return base_info + f" | {v} VIOLATION(S) FOUND: {self.violation_labels}"

        return base_info


# ---------------------------------------------------------------------------
# Base analyzer
# ---------------------------------------------------------------------------
class BaseAnalyzer(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def analyze(self, image_source: "str | Path | np.ndarray") -> AnalysisResult:
        """
        Accetta un percorso file (str/Path) oppure un frame già in memoria
        (np.ndarray BGR, come restituito da cv2.VideoCapture).
        Nessun file temporaneo viene scritto su disco.
        """
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
    def _run_inference(self, image_source: "str | np.ndarray") -> list[Detection]: ...

    # ── Helpers per VLM che hanno bisogno di bytes/base64 ────────────────────

    @staticmethod
    def _to_bgr(image_source: "str | Path | np.ndarray") -> "np.ndarray":
        """Restituisce sempre un array BGR, da path o da ndarray già in memoria."""
        import numpy as np
        if isinstance(image_source, np.ndarray):
            return image_source
        img = cv.imread(str(image_source))
        if img is None:
            raise ValueError(f"Impossibile leggere l'immagine: {image_source}")
        return img

    @staticmethod
    def _to_pil(image_source: "str | Path | np.ndarray"):
        """Restituisce una PIL Image RGB, da path o da ndarray BGR."""
        from PIL import Image
        import numpy as np
        if isinstance(image_source, np.ndarray):
            return Image.fromarray(cv.cvtColor(image_source, cv.COLOR_BGR2RGB))
        return Image.open(str(image_source)).convert("RGB")

    @staticmethod
    def _to_base64(image_source: "str | Path | np.ndarray") -> tuple[str, str]:
        """
        Ritorna (base64_string, media_type).
        Se ndarray: encode in-memory come JPEG (nessun file su disco).
        Se path: legge direttamente dal file.
        """
        import numpy as np
        if isinstance(image_source, np.ndarray):
            ok, buf = cv.imencode(".jpg", image_source)
            if not ok:
                raise RuntimeError("Encoding JPEG fallito")
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            return b64, "image/jpeg"
        path = str(image_source)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(path).suffix.lower().lstrip(".")
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                      "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return b64, media_type


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

    def _run_inference(self, image_source: "str | np.ndarray") -> list[Detection]:
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


# ---------------------------------------------------------------------------
# Foundry GPT-4o Analyzer
# ---------------------------------------------------------------------------
_SAFETY_SYSTEM_PROMPT = """You are a construction site safety inspector AI.
Analyze the image and detect PPE (Personal Protective Equipment) violations and unsafe behaviors.

For each person visible, check for:
- Helmet/hard hat (missing = no_helmet violation)
- High-visibility vest or jacket (missing = no_vest violation)
- Safety glove (missing = no_glove violation) (check each hand)
- Safety boot (missing = no_boot violation) (check each foot)
- Any unsafe posture, action, or presence in restricted areas (= unsafe_behavior)

Return one single detection for each hand and foot.
Respond ONLY with a JSON array. Each element has:
{
  "label": "<one of: no_helmet, no_vest, no_glove, no_boot, unsafe_behavior, helmet, vest, glove, boot, person>",
  "confidence": <float 0.0-1.0>,
  "description": "<brief description>"
}

If no people or violations are found, return an empty array [].
Return ONLY the JSON, no markdown fences, no explanation."""


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

    def _run_inference(self, image_source: "str | np.ndarray") -> list[Detection]:
        if not self.api_key:
            msg = "AZURE_OPENAI_KEY not set"
            raise RuntimeError(msg)

        image_data, media_type = self._to_base64(image_source)

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
# Florence-2 Local Analyzer (Validator)
# ---------------------------------------------------------------------------
class Florence2Analyzer(BaseAnalyzer):
    """
    Small-VLM local validator.
    Use this to confirm YOLO detections using natural language prompts.
    Requires: transformers, einops, timm
    """

    def __init__(self, model_id: str = "microsoft/Florence-2-base", device: str = "cpu"):
        super().__init__("Florence-2-Validator")
        self.device = device
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).to(self.device).eval()

    def _run_inference(self, image_source: "str | np.ndarray") -> list[Detection]:
        self._load_model()
        img = self._to_bgr(image_source)[:, :, ::-1]  # BGR→RGB

        # Prompt espanso per includere tutto il vocabolario necessario alla validazione
        prompt = "<CAPTION_TO_PHRASE_GROUNDING> person, safety helmet, high visibility vest, safety gloves, bare arms"
        prompt = "<DETAILED_CAPTION> person visible from waist up, checking for helmet and vest"

        inputs = self._processor(text=prompt, images=img, return_tensors="pt").to(self.device)

        # Generazione (utilizziamo num_beams per una maggiore precisione nel grounding)
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )

        results = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self._processor.post_process_generation(
            results, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(img.shape[1], img.shape[0])
        )

        detections = []
        data = parsed_answer.get("<CAPTION_TO_PHRASE_GROUNDING>", {})

        for label, bboxes in zip(data.get("labels", []), data.get("bboxes", []), strict=False):
            # Normalizzazione etichette per il confronto
            clean_label = label.lower().strip().replace(" ", "_")
            # Florence-2 restituisce [x1, y1, x2, y2], convertiamo in 4 punti se necessario per il tuo canvas
            detections.append(Detection(label=clean_label, confidence=0.85, bbox=bboxes))

        return detections


# ---------------------------------------------------------------------------
# Moondream2 Local VLM Analyzer (Prompt-based)
# ---------------------------------------------------------------------------
class MoondreamAnalyzer(BaseAnalyzer):
    """
    Tiny VLM for local inference. Supports natural language prompting.
    Perfect for validating YOLO detections with complex safety logic.
    Requires: transformers, einops, pillow
    """

    def __init__(self, model_id: str = "vikhyatk/moondream2", revision: str = "2025-06-21", device: str = "cpu"):
        super().__init__("Moondream2-Local")
        self.device = device
        self.model_id = model_id
        self.revision = revision
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Carichiamo il tokenizer e il modello
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self._model = (
            AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=True)
            .to(self.device)
            .eval()
        )

    def _run_inference(self, image_source: "str | np.ndarray") -> list[Detection]:
        self._load_model()
        image = self._to_pil(image_source)
        enc_image = self._model.encode_image(image)

        # Moondream ha un metodo dedicato per rispondere alle domande
        answer = self._model.answer_question(enc_image, _SAFETY_SYSTEM_PROMPT, self._tokenizer)

        detections = []
        try:
            # Pulizia minima se il modello aggiunge chiacchiere
            clean_answer = answer.strip()
            if "```json" in clean_answer:
                clean_answer = clean_answer.split("```json")[1].split("```")[0]

            items = json.loads(clean_answer)
            if isinstance(items, dict):
                items = [items]  # se restituisce un solo oggetto

            for item in items:
                detections.append(
                    Detection(
                        label=item.get("label", "unknown"),
                        confidence=float(item.get("confidence", 0.7)),
                        bbox=None,  # Moondream2 standard non restituisce bbox precisi come Florence
                    )
                )
        except:
            # Se il parsing fallisce, facciamo un controllo testuale semplice
            if "no_helmet" in answer.lower():
                detections.append(Detection(label="no_helmet", confidence=0.8))

        return detections


# ---------------------------------------------------------------------------
# Temporal Persistence Tracker
# ---------------------------------------------------------------------------
class PersistenceTracker:
    """
    Tracks violations over time to reduce false positives.
    An alert is only triggered if a violation persists for N frames.
    """

    def __init__(self, threshold_frames: int = 5):
        self.threshold_frames = threshold_frames
        self.active_violations = {}  # {label: count}

    def update(self, current_violations: list[str]) -> list[str]:
        confirmed = []
        # Incrementa contatori per violazioni attuali
        for v in current_violations:
            self.active_violations[v] = self.active_violations.get(v, 0) + 1
            if self.active_violations[v] >= self.threshold_frames:
                confirmed.append(v)

        # Decrementa o rimuovi violazioni non viste in questo frame
        for v in list(self.active_violations.keys()):
            if v not in current_violations:
                self.active_violations[v] -= 1
                if self.active_violations[v] <= 0:
                    del self.active_violations[v]

        return confirmed


# ---------------------------------------------------------------------------
# Hybrid Cascade Pipeline
# ---------------------------------------------------------------------------
# Mappa la violazione di YOLO al DPI positivo che Florence-2 dovrebbe trovare
CROSS_VALIDATION_MAP = {
    "Non-Helmet": "safety_helmet",
    "Non-Vest": "safety_vest",  # Se YOLO avesse Non-Vest
    "Bare-arm": "arm",  # Per verificare se la pelle è scoperta
    "Glove": "glove",  # Per confermare la presenza
}


class SafetyHybridPipeline:
    def __init__(self, primary_yolo: YOLOAnalyzer, validator_vlm: Florence2Analyzer):
        self.yolo = primary_yolo
        self.validator = validator_vlm
        # Soglia di persistenza: 10 frame di fila prima di chiamare il VLM
        self.tracker = PersistenceTracker(threshold_frames=10)

    def analyze_frame(self, image_path: str) -> dict:
        # Step 1: YOLO Inference
        yolo_res = self.yolo.analyze(image_path)

        # Estraiamo i nomi delle violazioni rilevate (es. "Non-Helmet", "Bare-arm")
        current_violations = [d.label for d in yolo_res.violations]

        # Step 2: Temporal Filtering (Evitiamo glitch momentanei)
        confirmed_by_tracker = self.tracker.update(current_violations)

        final_result = {"yolo_detections": yolo_res.detections, "status": "SAFE", "validated_violations": [], "alerts": []}

        # Step 3: Validazione con Florence-2
        if confirmed_by_tracker:
            print(f"[*] Validating persistent violations: {confirmed_by_tracker}...")
            vlm_res = self.validator.analyze(image_path)
            vlm_labels = [d.label for d in vlm_res.detections]

            for v_label in confirmed_by_tracker:
                # Esempio logico: se YOLO vede "Non-Helmet"
                # Florence-2 DEVE trovare "safety_helmet" per dire che è SAFE.
                # Se NON lo trova, confermiamo l'allerta.

                if v_label == "Non-Helmet":
                    if "safety_helmet" not in vlm_labels:
                        final_result["validated_violations"].append("MISSING_HELMET")
                        final_result["status"] = "ALERT"

                elif v_label == "Vest":  # Se YOLO fosse incerto
                    if "safety_vest" in vlm_labels:
                        # Conferma positiva
                        pass

                # Aggiungi qui altre logiche per Bare-arm o Glove...

        return final_result


# ---------------------------------------------------------------------------
# First Pipeline
# ---------------------------------------------------------------------------
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
