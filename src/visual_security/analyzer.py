"""
Safety analyzer for construction site PPE detection.

Detection backend: modelli **open-vocabulary** con licenza Apache 2.0,
nativi in `transformers` — nessuna dipendenza Ultralytics/YOLO (AGPL,
incompatibile con codice proprietario in produzione).

Backend disponibili (stessa interfaccia, si sceglie con `build_detector`):

  - GroundingDinoAnalyzer  (default)
      IDEA-Research/grounding-dino-base — Apache 2.0.
      Il riferimento per accuratezza zero-shot: rileva qualunque classe
      descritta a testo ("a hard hat", "a safety vest", ...) SENZA training.
      Piu' lento (encoder testo+immagine con fusione profonda).

  - OmDetTurboAnalyzer
      omlab/omdet-turbo-swin-tiny-hf — Apache 2.0.
      Open-vocabulary real-time: zero-shot forte (supera Grounding-DINO-L
      su LVIS) e text-embedding cache-abili (il prompt qui e' fisso).
      Molto piu' veloce, leggermente meno accurato sugli oggetti piccoli.

Perche' open-vocabulary e non un detector COCO (RT-DETR/D-FINE):
i COCO-pretrained conoscono "person" ma NON caschi/gilet/guanti — servirebbe
fine-tuning su un dataset PPE, che e' esattamente il collo di bottiglia
storico del progetto. I modelli open-vocabulary rilevano le classi PPE
direttamente dal prompt testuale, zero-shot.
"""

from __future__ import annotations

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
PPE_LABELS = ["Glove", "Helmet", "Person", "Shoe", "Vest"]

# Prompt testuali per il detector open-vocabulary -> categoria canonica.
# Frasi brevi e concrete funzionano meglio delle categorie astratte
# ("hard hat" > "helmet" per i caschi da cantiere).
DETECTION_PROMPTS: dict[str, str] = {
    "a person": "Person",
    "a hard hat": "Helmet",
    "a reflective safety vest": "Vest",
    "a work glove": "Glove",
    "a work boot": "Shoe",
}

# Parole chiave per riportare il testo matchato dal modello alla categoria
# canonica (Grounding DINO puo' restituire sotto-frasi del prompt).
_KEYWORD_TO_LABEL: list[tuple[str, str]] = [
    ("person", "Person"),
    ("hard hat", "Helmet"),
    ("helmet", "Helmet"),
    ("vest", "Vest"),
    ("glove", "Glove"),
    ("boot", "Shoe"),
    ("shoe", "Shoe"),
]


def _match_label(text: str) -> str | None:
    t = text.lower()
    for kw, label in _KEYWORD_TO_LABEL:
        if kw in t:
            return label
    return None


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[float] | None = None  # [x1, y1, x2, y2] pixel assoluti
    is_violation: bool = field(default=False)


@dataclass
class AnalysisResult:
    model_name: str
    image_path: str
    detections: list[Detection]
    inference_time_ms: float
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return f"[{self.model_name}] ERROR: {self.error}"
        det_list = ", ".join([f"{d.label}({d.confidence:.2f})" for d in self.detections])
        return f"[{self.model_name}] {len(self.detections)} detections ({det_list}) in {self.inference_time_ms:.1f}ms"


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
        except Exception as e:  # noqa: BLE001 — la pipeline degrada, non crasha
            detections = []
            error = f"{type(e).__name__}: {e}"
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
# Open-vocabulary analyzers (transformers, Apache 2.0)
# ---------------------------------------------------------------------------
class _OpenVocabAnalyzer(BaseAnalyzer):
    """Base comune: caricamento pigro, conversione PIL, device auto."""

    def __init__(
        self,
        model_name: str,
        model_id: str,
        conf_threshold: float,
        prompts: dict[str, str] | None = None,
        device: str | None = None,
    ):
        super().__init__(model_name)
        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.prompts = prompts or dict(DETECTION_PROMPTS)
        self.device = device
        self._model = None
        self._processor = None

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _to_pil(image_bgr: np.ndarray):
        from PIL import Image

        return Image.fromarray(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB))

    @abstractmethod
    def _load(self) -> None: ...


class GroundingDinoAnalyzer(_OpenVocabAnalyzer):
    """
    Zero-shot detection con Grounding DINO (Apache 2.0, nativo transformers).

    Il prompt e' la concatenazione delle frasi in `prompts` separate da ". "
    (formato richiesto dal modello: minuscolo, frasi separate da punto).
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        conf_threshold: float = 0.35,
        text_threshold: float = 0.25,
        prompts: dict[str, str] | None = None,
        device: str | None = None,
    ):
        super().__init__("GroundingDINO", model_id, conf_threshold, prompts, device)
        self.text_threshold = text_threshold

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = self._resolve_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, dtype=dtype).to(device).eval()
        self._text_prompt = ". ".join(p.lower() for p in self.prompts) + "."

    def _run_inference(self, image_source: str | np.ndarray) -> list[Detection]:
        import torch

        self._load()
        img = self._to_bgr(image_source)
        pil = self._to_pil(img)

        inputs = self._processor(images=pil, text=self._text_prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=self.conf_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],  # (h, w)
        )[0]

        # transformers >=4.51 usa "text_labels"; le versioni precedenti "labels".
        texts = results.get("text_labels", results.get("labels"))
        detections: list[Detection] = []
        for box, score, text in zip(results["boxes"], results["scores"], texts):
            label = _match_label(str(text))
            if label is None:
                continue
            detections.append(
                Detection(label=label, confidence=float(score), bbox=[float(v) for v in box.tolist()])
            )
        return detections


class OmDetTurboAnalyzer(_OpenVocabAnalyzer):
    """
    Zero-shot detection real-time con OmDet-Turbo (Apache 2.0, nativo transformers).

    Alternativa veloce a Grounding DINO: text-embedding cache-abili e
    inferenza molto piu' rapida, a costo di un po' di accuratezza sugli
    oggetti piccoli.
    """

    def __init__(
        self,
        model_id: str = "omlab/omdet-turbo-swin-tiny-hf",
        conf_threshold: float = 0.30,
        prompts: dict[str, str] | None = None,
        device: str | None = None,
    ):
        super().__init__("OmDet-Turbo", model_id, conf_threshold, prompts, device)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, OmDetTurboForObjectDetection

        device = self._resolve_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = OmDetTurboForObjectDetection.from_pretrained(self.model_id, dtype=dtype).to(device).eval()
        self._classes = list(self.prompts)

    def _run_inference(self, image_source: str | np.ndarray) -> list[Detection]:
        import torch

        self._load()
        img = self._to_bgr(image_source)
        pil = self._to_pil(img)

        inputs = self._processor(images=pil, text=self._classes, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # La firma del post-processing e' cambiata tra versioni di transformers
        # (classes= -> text_labels=): proviamo la nuova, ripieghiamo sulla vecchia.
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                text_labels=[self._classes],
                threshold=self.conf_threshold,
                target_sizes=[pil.size[::-1]],
            )[0]
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                classes=[self._classes],
                score_threshold=self.conf_threshold,
                target_sizes=[pil.size[::-1]],
            )[0]

        texts = results.get("text_labels", results.get("classes"))
        detections: list[Detection] = []
        for box, score, text in zip(results["boxes"], results["scores"], texts):
            label = self.prompts.get(str(text)) or _match_label(str(text))
            if label is None:
                continue
            detections.append(
                Detection(label=label, confidence=float(score), bbox=[float(v) for v in box.tolist()])
            )
        return detections


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
DETECTOR_CHOICES = ("grounding-dino", "omdet-turbo")


def build_detector(
    detector: str = "grounding-dino",
    conf_threshold: float | None = None,
    device: str | None = None,
) -> BaseAnalyzer:
    """
    Crea il detector open-vocabulary.

    Parameters
    ----------
    detector : str
        "grounding-dino" (default, massima accuratezza) o
        "omdet-turbo" (real-time, leggermente meno accurato).
    conf_threshold : float | None
        Soglia confidence; None = default del backend.
    """
    if detector == "grounding-dino":
        kwargs = {"conf_threshold": conf_threshold} if conf_threshold is not None else {}
        return GroundingDinoAnalyzer(device=device, **kwargs)
    if detector == "omdet-turbo":
        kwargs = {"conf_threshold": conf_threshold} if conf_threshold is not None else {}
        return OmDetTurboAnalyzer(device=device, **kwargs)
    msg = f"Detector sconosciuto: {detector!r}. Scelte: {DETECTOR_CHOICES}"
    raise ValueError(msg)
