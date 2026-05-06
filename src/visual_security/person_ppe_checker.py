"""
Person-centric PPE Completeness Checker.

Per ogni 'Person' rilevata da YOLO, verifica che i PPE richiesti
(Helmet, Vest, Glove ×2, Shoe ×2) siano spazialmente associati
tramite containment overlap con la bounding box della persona.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REQUIRED_PPE_COUNTS: dict[str, int] = {
    "Helmet": 1,
    "Vest": 1,
    "Glove": 2,
    "Shoe": 2,
}

_PPE_LABEL_MAP: dict[str, str] = {
    "helmet": "Helmet",
    "hardhat": "Helmet",
    "hard-hat": "Helmet",
    "hard_hat": "Helmet",
    "vest": "Vest",
    "hi-vis": "Vest",
    "hiviz": "Vest",
    "safety-vest": "Vest",
    "glove": "Glove",
    "gloves": "Glove",
    "shoe": "Shoe",
    "shoes": "Shoe",
    "boot": "Shoe",
    "boots": "Shoe",
    "safety-boot": "Shoe",
    "safety-boots": "Shoe",
}


def _to_xyxy(bbox, frame_w: int = 640, frame_h: int = 640) -> tuple[float, float, float, float] | None:
    """
    Normalizza qualsiasi formato bbox a (x1, y1, x2, y2) in pixel assoluti.

    Formati supportati:
      [x1, y1, x2, y2]  pixel assoluti         → passthrough
      [x1, y1, x2, y2]  normalizzati (0-1)     → moltiplica per w,h
      [cx, cy, w, h]    pixel assoluti (center) → converti
      [cx, cy, w, h]    normalizzati (0-1)      → converti + scala
      [[x1,y1],[x2,y2],...]  polygon            → axis-aligned bbox
    """
    if bbox is None:
        return None
    try:
        first = bbox[0]
        if isinstance(first, (list, tuple)):
            # Polygon: lista di punti
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            return (min(xs), min(ys), max(xs), max(ys))

        coords = [float(v) for v in bbox]

        if len(coords) == 4:
            a, b, c, d = coords

            # Rilevazione automatica del formato
            # Se tutti i valori sono in [0,1] → normalizzati
            if all(0.0 <= v <= 1.0 for v in coords):
                # Potrebbe essere [x1,y1,x2,y2] norm o [cx,cy,w,h] norm
                # Euristica: se c < a o d < b → center format
                if c < a or d < b:
                    # [cx, cy, w, h] normalizzato
                    x1 = (a - c / 2) * frame_w
                    y1 = (b - d / 2) * frame_h
                    x2 = (a + c / 2) * frame_w
                    y2 = (b + d / 2) * frame_h
                else:
                    # [x1, y1, x2, y2] normalizzato
                    x1, y1, x2, y2 = a * frame_w, b * frame_h, c * frame_w, d * frame_h
                return (x1, y1, x2, y2)

            # Valori pixel: potrebbe essere xyxy o cxcywh
            # Se c < a o d < b sicuramente è center format
            if c < a or d < b:
                # [cx, cy, w, h] pixel
                x1, y1 = a - c / 2, b - d / 2
                x2, y2 = a + c / 2, b + d / 2
            else:
                # [x1, y1, x2, y2] pixel — formato più comune
                x1, y1, x2, y2 = a, b, c, d

            return (x1, y1, x2, y2)

        if len(coords) >= 8:
            xs, ys = coords[0::2], coords[1::2]
            return (min(xs), min(ys), max(xs), max(ys))

    except (TypeError, ValueError, IndexError):
        pass
    return None


def _containment(inner: tuple, outer: tuple) -> float:
    """Frazione di *inner* coperta da *outer* (0–1)."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def _iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio(ppe_box: tuple, person_box: tuple) -> float:
    """
    Score composito: max tra containment e IoU.
    Containment è il metodo principale (PPE è piccolo, Person è grande).
    IoU come fallback per elementi grande-scala (es. Vest ≈ torso).
    """
    return max(_containment(ppe_box, person_box), _iou(ppe_box, person_box))


@dataclass
class PersonPPEResult:
    person_idx: int
    person_bbox: tuple | None
    person_conf: float
    frame_size: tuple = (640, 640)  # (w, h) usato per normalizzare bbox
    found_ppe: dict[str, int] = field(default_factory=dict)
    missing_ppe: list[str] = field(default_factory=list)
    associated_ppe: list = field(default_factory=list)
    needs_vlm_validation: bool = False

    @property
    def is_compliant(self) -> bool:
        return len(self.missing_ppe) == 0

    def summary(self) -> str:
        status = "✅ OK" if self.is_compliant else f"⚠️  MANCANTI: {self.missing_ppe}"
        return f"Persona#{self.person_idx}(conf={self.person_conf:.2f}) trovati={self.found_ppe} | {status}" + (
            " [VLM→]" if self.needs_vlm_validation else ""
        )


class PersonPPEChecker:
    """
    Associa le detection YOLO alle persone e valuta la completezza dei PPE.

    Parameters
    ----------
    required_ppe : dict[str, int]
        Quantità minima per ogni PPE. Default = set completo.
    containment_threshold : float
        Soglia minima di overlap per associare un PPE a una persona. Default 0.25.
        Abbassata rispetto alla versione precedente per essere meno restrittiva
        con soggetti parzialmente inquadrati.
    iou_threshold : float
        Soglia IoU fallback. Default 0.05.
    no_bbox_policy : str
        "vlm_only"    → persona senza bbox: tutti i PPE mancanti, VLM obbligatorio
        "none_missing" → persona senza bbox: considerata compliant (conservativo)
    """

    FULL_PPE = REQUIRED_PPE_COUNTS

    def __init__(
        self,
        required_ppe: dict[str, int] | None = None,
        containment_threshold: float = 0.25,
        iou_threshold: float = 0.05,
        no_bbox_policy: str = "vlm_only",
        vlm_trigger_missing: set[str] | None = None,
    ):
        self.required_ppe = required_ppe or dict(REQUIRED_PPE_COUNTS)
        self.containment_threshold = containment_threshold
        self.iou_threshold = iou_threshold
        self.no_bbox_policy = no_bbox_policy
        self.vlm_trigger_missing = vlm_trigger_missing

    def check(self, detections: list, frame_w: int = 640, frame_h: int = 640) -> list[PersonPPEResult]:
        """
        Ritorna un PersonPPEResult per ogni Person rilevata.
        frame_w/frame_h servono per denormalizzare bbox se necessario.
        """
        persons = [(i, d) for i, d in enumerate(detections) if d.label.lower() == "person"]
        ppe_items = [
            (i, d, _PPE_LABEL_MAP[d.label.lower().strip()])
            for i, d in enumerate(detections)
            if d.label.lower().strip() in _PPE_LABEL_MAP
        ]

        if not persons:
            return []

        # Pre-calcola le bbox di tutte le persone
        person_boxes = {i: _to_xyxy(d.bbox, frame_w, frame_h) for i, d in persons}

        # ── Greedy assignment: ogni PPE va alla persona con overlap maggiore ──
        # Evita che uno stesso item venga contato per più persone
        assignments: dict[int, list[tuple[str, object]]] = {i: [] for i, _ in persons}

        for _, item_det, ppe_cat in ppe_items:
            item_box = _to_xyxy(item_det.bbox, frame_w, frame_h)
            if item_box is None:
                continue

            best_score = -1.0
            best_pidx = None
            for p_i, _ in persons:
                p_box = person_boxes.get(p_i)
                if p_box is None:
                    continue
                score = _overlap_ratio(item_box, p_box)
                if score >= self.containment_threshold and score > best_score:
                    best_score = score
                    best_pidx = p_i

            if best_pidx is not None:
                assignments[best_pidx].append((ppe_cat, item_det))

        # ── Costruisci i risultati ─────────────────────────────────────────────
        results: list[PersonPPEResult] = []
        for p_i, p_det in persons:
            p_box = person_boxes[p_i]

            found_counts: dict[str, int] = dict.fromkeys(self.required_ppe, 0)
            assoc_dets = []
            for ppe_cat, det in assignments[p_i]:
                found_counts[ppe_cat] = found_counts.get(ppe_cat, 0) + 1
                assoc_dets.append(det)

            if p_box is None:
                if self.no_bbox_policy == "vlm_only":
                    missing, needs_vlm = list(self.required_ppe.keys()), True
                else:
                    missing, needs_vlm = [], False
            else:
                missing = [cat for cat, req in self.required_ppe.items() if found_counts.get(cat, 0) < req]
                needs_vlm = (
                    bool(missing) if self.vlm_trigger_missing is None else bool(set(missing) & self.vlm_trigger_missing)
                )

            results.append(
                PersonPPEResult(
                    person_idx=p_i,
                    person_bbox=p_box,
                    person_conf=p_det.confidence,
                    frame_size=(frame_w, frame_h),
                    found_ppe=found_counts,
                    missing_ppe=missing,
                    associated_ppe=assoc_dets,
                    needs_vlm_validation=needs_vlm,
                )
            )

        return results
