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
PPE_LABELS = ["Cigarette", "Glasses", "Glove", "Helmet", "Person", "Shoe", "Vest"]

# Prompt testuali per il detector open-vocabulary -> categoria canonica.
# Frasi brevi e concrete funzionano meglio delle categorie astratte
# ("hard hat" > "helmet" per i caschi da cantiere).
#
# "Glasses" e' un PPE richiesto (come casco/gilet); "Cigarette" e' invece un
# item PROIBITO: la sua presenza addosso a una persona e' una violazione (vedi
# person_ppe_checker.PROHIBITED_ITEMS). Il detector li rileva allo stesso modo,
# e' il checker a distinguere "manca" da "non deve esserci".
DETECTION_PROMPTS: dict[str, str] = {
    "a person": "Person",
    "a hard hat": "Helmet",
    "a reflective safety vest": "Vest",
    "safety glasses": "Glasses",
    "a work glove": "Glove",
    "a work boot": "Shoe",
    "a cigarette": "Cigarette",
}

# Soglie di confidence PER-CLASSE (floor minimo per accettare una detection).
# Le classi non elencate usano la soglia base del detector (`conf_threshold`).
# Motivazione: occhiali e sigarette sono classi piccole/difficili per i detector
# open-vocabulary e generano falsi positivi a bassa confidence. Misurato su
# ppe_video.mp4: i FP di Glasses hanno tetto ~0.42 (GD) / ~0.29 (OmDet) -> soglia
# 0.45 li azzera. I FP di Cigarette arrivano fino a ~0.49 su GD (outlier isolato)
# -> soglia 0.50. Entrambe restano ben sotto le classi reali (Person/Helmet/Vest
# fino a 0.8-0.9), quindi il filtro e' selettivo.
# NB: valori tarati su QUESTO video (dove occhiali/sigarette non esistono davvero);
# se in un video reale servisse rilevare occhiali/sigarette veri, abbassare qui.
DETECTION_CONF: dict[str, float] = {
    "Glasses": 0.45,
    "Cigarette": 0.50,
}

# OWLv2 usa score sigmoidi su una scala DIVERSA da Grounding DINO/OmDet, quindi
# ha soglie per-classe proprie. Valori derivati dallo sweep F1 su SH17 (vedi
# evaluation/REPORT.md). NB: provvisori per il video — non ancora tarati sui
# falsi positivi di un video reale come per gli altri due backend.
OWLV2_CONF: dict[str, float] = {
    "Person": 0.15,
    "Helmet": 0.35,
    "Vest": 0.30,
    "Glasses": 0.30,
    "Glove": 0.25,
    "Shoe": 0.15,
    "Cigarette": 0.35,
}

# Parole chiave per riportare il testo matchato dal modello alla categoria
# canonica (Grounding DINO puo' restituire sotto-frasi del prompt).
_KEYWORD_TO_LABEL: list[tuple[str, str]] = [
    ("person", "Person"),
    ("hard hat", "Helmet"),
    ("helmet", "Helmet"),
    ("vest", "Vest"),
    ("glasses", "Glasses"),
    ("goggle", "Glasses"),
    ("glove", "Glove"),
    ("boot", "Shoe"),
    ("shoe", "Shoe"),
    ("cigarette", "Cigarette"),
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


def _box_iou(a: list[float] | None, b: list[float] | None) -> float:
    """IoU tra due bbox [x1,y1,x2,y2]. 0 se una manca."""
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_per_class(detections: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """
    Non-Maximum Suppression **per-classe**: rimuove i box duplicati/sovrapposti
    della STESSA classe (IoU > iou_thr), tenendo quello a confidence maggiore.
    I detector open-vocabulary tendono a emettere più box sullo stesso oggetto;
    senza NMS questo produce conteggi gonfiati (es. "Shoe 6") e falsi positivi.
    Classi diverse non si sopprimono (una persona e il suo casco si sovrappongono
    ma sono entrambi validi).
    """
    from collections import defaultdict

    by_label: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        by_label[d.label].append(d)

    kept: list[Detection] = []
    for group in by_label.values():
        for d in sorted(group, key=lambda x: -x.confidence):
            if all(_box_iou(d.bbox, k.bbox) <= iou_thr for k in kept if k.label == d.label):
                kept.append(d)
    return kept


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
        class_conf: dict[str, float] | None = None,
        nms_iou: float = 0.5,
    ):
        super().__init__(model_name)
        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.prompts = prompts or dict(DETECTION_PROMPTS)
        self.class_conf = class_conf if class_conf is not None else dict(DETECTION_CONF)
        # IoU per la NMS per-classe applicata a fine inferenza. None/<=0 = disattivata.
        self.nms_iou = nms_iou
        self.device = device
        self._model = None
        self._processor = None

    def _accept(self, label: str, score: float) -> bool:
        """True se lo score supera la soglia per-classe (o quella base)."""
        return score >= self.class_conf.get(label, self.conf_threshold)

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
            if label is None or not self._accept(label, float(score)):
                continue
            detections.append(
                Detection(label=label, confidence=float(score), bbox=[float(v) for v in box.tolist()])
            )
        return nms_per_class(detections, self.nms_iou) if self.nms_iou and self.nms_iou > 0 else detections


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
            if label is None or not self._accept(label, float(score)):
                continue
            detections.append(
                Detection(label=label, confidence=float(score), bbox=[float(v) for v in box.tolist()])
            )
        return nms_per_class(detections, self.nms_iou) if self.nms_iou and self.nms_iou > 0 else detections


class Owlv2Analyzer(_OpenVocabAnalyzer):
    """
    Zero-shot detection con OWLv2 (google/owlv2, Apache 2.0, nativo transformers).

    Alternativa a Grounding DINO/OmDet valutata nel benchmark: quasi pari a
    Grounding DINO in accuratezza zero-shot con meno calcolo. I "text queries"
    sono le frasi dei prompt; le label dei risultati sono indici in quella lista.
    Gli score sono sigmoidi (bassi): la soglia di default e' piu' bassa (0.10).
    """

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        conf_threshold: float = 0.10,
        prompts: dict[str, str] | None = None,
        device: str | None = None,
        class_conf: dict[str, float] | None = None,
    ):
        # Default alle soglie per-classe di OWLv2 (scala score diversa).
        super().__init__(
            "OWLv2", model_id, conf_threshold, prompts, device,
            class_conf=class_conf if class_conf is not None else dict(OWLV2_CONF),
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        device = self._resolve_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._processor = Owlv2Processor.from_pretrained(self.model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id, dtype=dtype).to(device).eval()
        self._classes = list(self.prompts)

    def _run_inference(self, image_source: str | np.ndarray) -> list[Detection]:
        import torch

        self._load()
        img = self._to_bgr(image_source)
        pil = self._to_pil(img)

        inputs = self._processor(text=[self._classes], images=pil, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]], device=self._model.device)  # (h, w)
        # La firma del post-processing e' cambiata tra versioni di transformers.
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=self.conf_threshold
            )[0]
        except (TypeError, AttributeError):
            results = self._processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=self.conf_threshold
            )[0]

        detections: list[Detection] = []
        for box, score, lab in zip(results["boxes"], results["scores"], results["labels"]):
            phrase = self._classes[int(lab)]
            label = self.prompts.get(phrase) or _match_label(phrase)
            if label is None or not self._accept(label, float(score)):
                continue
            detections.append(
                Detection(label=label, confidence=float(score), bbox=[float(v) for v in box.tolist()])
            )
        return nms_per_class(detections, self.nms_iou) if self.nms_iou and self.nms_iou > 0 else detections


class EnsembleAnalyzer(BaseAnalyzer):
    """
    Ensemble di due detector: usa il `primary` per tutte le classi tranne quelle
    in `secondary_classes`, per cui usa il `secondary`. Nato per coprire il punto
    debole di OWLv2 (occhiali) con Grounding DINO, che sugli occhiali e' migliore.

    Costo = somma delle due inferenze (accuracy-first, non real-time).
    """

    def __init__(self, primary: BaseAnalyzer, secondary: BaseAnalyzer, secondary_classes: set[str]):
        super().__init__(f"Ensemble({primary.model_name}+{secondary.model_name})")
        self.primary = primary
        self.secondary = secondary
        self.secondary_classes = set(secondary_classes)
        self.model_id = f"{getattr(primary, 'model_id', '?')} + {getattr(secondary, 'model_id', '?')} [{'/'.join(sorted(secondary_classes))}]"

    def _run_inference(self, image_source: str | np.ndarray) -> list[Detection]:
        p = self.primary.analyze(image_source)
        s = self.secondary.analyze(image_source)
        dets = [d for d in p.detections if d.label not in self.secondary_classes]
        dets += [d for d in s.detections if d.label in self.secondary_classes]
        return dets


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
DETECTOR_CHOICES = ("grounding-dino", "omdet-turbo", "owlv2", "ensemble")


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
    if detector == "owlv2":
        kwargs = {"conf_threshold": conf_threshold} if conf_threshold is not None else {}
        return Owlv2Analyzer(device=device, **kwargs)
    if detector == "ensemble":
        # OWLv2 (piu' accurato) per tutte le classi, Grounding DINO solo per gli
        # occhiali (dove OWLv2 e' debole). Accuracy-first, lento (somma dei due).
        return EnsembleAnalyzer(
            primary=Owlv2Analyzer(device=device),
            secondary=GroundingDinoAnalyzer(device=device),
            secondary_classes={"Glasses"},
        )
    msg = f"Detector sconosciuto: {detector!r}. Scelte: {DETECTOR_CHOICES}"
    raise ValueError(msg)
