"""
Person-centric PPE Completeness Checker.

Per ogni 'Person' rilevata dal detector, verifica due cose associando gli
oggetti alla bounding box della persona tramite containment overlap:

  * PPE RICHIESTI (Helmet, Vest, Glasses, Glove ×2, Shoe ×2): devono essere
    presenti; se mancano -> violazione.
  * item PROIBITI (Cigarette): NON devono essere presenti; se sono associati
    a una persona -> violazione (logica invertita rispetto ai PPE).
"""

from __future__ import annotations

from dataclasses import dataclass, field

REQUIRED_PPE_COUNTS: dict[str, int] = {
    "Helmet": 1,
    "Vest": 1,
    "Glasses": 1,
    "Glove": 2,
    "Shoe": 2,
}

# Item la cui PRESENZA addosso a una persona costituisce violazione.
PROHIBITED_ITEMS: tuple[str, ...] = ("Cigarette",)

_PPE_LABEL_MAP: dict[str, str] = {
    "helmet": "Helmet",
    "hardhat": "Helmet",
    "hard-hat": "Helmet",
    "hard_hat": "Helmet",
    "vest": "Vest",
    "hi-vis": "Vest",
    "hiviz": "Vest",
    "safety-vest": "Vest",
    "glasses": "Glasses",
    "goggles": "Glasses",
    "safety-glasses": "Glasses",
    "glove": "Glove",
    "gloves": "Glove",
    "shoe": "Shoe",
    "shoes": "Shoe",
    "boot": "Shoe",
    "boots": "Shoe",
    "safety-boot": "Shoe",
    "safety-boots": "Shoe",
}

_PROHIBITED_LABEL_MAP: dict[str, str] = {
    "cigarette": "Cigarette",
    "cigarettes": "Cigarette",
    "cig": "Cigarette",
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
    # Item proibiti (es. sigarette) trovati addosso alla persona: la loro
    # presenza e' gia' di per se' una violazione.
    prohibited_present: list[str] = field(default_factory=list)
    associated_ppe: list = field(default_factory=list)
    # Identita' persistente assegnata da PersonTracker (None = non tracciata,
    # es. analisi single-frame in debug_frame.py).
    track_id: int | None = None

    @property
    def is_compliant(self) -> bool:
        return not self.missing_ppe and not self.prohibited_present

    @property
    def violation_labels(self) -> list[str]:
        """
        Etichette di tutte le violazioni della persona, usate come chiave di
        persistenza / testo negli alert: PPE mancanti + item proibiti presenti
        (questi ultimi prefissati "NO " per distinguerli, es. "NO Cigarette").
        """
        return [*self.missing_ppe, *(f"NO {p}" for p in self.prohibited_present)]

    def summary(self) -> str:
        if self.is_compliant:
            status = "OK"
        else:
            parts = []
            if self.missing_ppe:
                parts.append(f"MANCANTI: {self.missing_ppe}")
            if self.prohibited_present:
                parts.append(f"VIETATI: {self.prohibited_present}")
            status = " ".join(parts)
        tid = f"[T{self.track_id}]" if self.track_id is not None else ""
        return f"Persona#{self.person_idx}{tid}(conf={self.person_conf:.2f}) trovati={self.found_ppe} | {status}"


class PersonPPEChecker:
    """
    Associa le detection alle persone e valuta DPI richiesti + item vietati.

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
        "all_missing"  → persona senza bbox: tutti i PPE marcati mancanti (prudente)
        "none_missing" → persona senza bbox: considerata compliant (conservativo)
    """

    FULL_PPE = REQUIRED_PPE_COUNTS

    def __init__(
        self,
        required_ppe: dict[str, int] | None = None,
        containment_threshold: float = 0.25,
        iou_threshold: float = 0.05,
        no_bbox_policy: str = "all_missing",
    ):
        self.required_ppe = required_ppe or dict(REQUIRED_PPE_COUNTS)
        self.containment_threshold = containment_threshold
        self.iou_threshold = iou_threshold
        self.no_bbox_policy = no_bbox_policy

    # ── Oggetti che tendono a stare FUORI dalla bbox persona ──────────────────
    # Gloves    → mani ai lati (espandi orizzontalmente)
    # Cigarette → in mano o alla bocca: come i guanti, spesso ai bordi laterali
    # Shoes     → piedi sotto (espandi verso il basso)
    # Helmet    → testa sopra (espandi verso l'alto)
    _EXPANDED_PPE = {"Glove", "Cigarette", "Shoe", "Helmet"}

    # ── Vincoli di plausibilita' geometrica per le classi soggette a FP ────────
    # Occhiali: stanno sul volto -> devono cadere nella fascia superiore della
    #   persona (entro _GLASSES_MAX_REL_Y dell'altezza, dall'alto).
    # Sigaretta: oggetto minuscolo e sottile -> area < _CIGARETTE_MAX_AREA_RATIO
    #   dell'area persona, larghezza < _CIGARETTE_MAX_WIDTH_RATIO della larghezza
    #   persona (una sigaretta larga 1/3 della persona e' un blob mal classificato),
    #   e non sta ai piedi.
    _GLASSES_MAX_REL_Y = 0.35
    _CIGARETTE_MAX_AREA_RATIO = 0.02
    _CIGARETTE_MAX_WIDTH_RATIO = 0.15
    _CIGARETTE_MAX_REL_Y = 0.75

    def _plausible(self, cat: str, item_box: tuple, p_box: tuple | None) -> bool:
        """
        Filtro anti-falsi-positivi per occhiali/sigarette: verifica che la
        posizione/dimensione dell'oggetto rispetto alla persona sia coerente
        con dove quell'oggetto puo' realmente trovarsi. Le altre classi passano.
        """
        if p_box is None or cat not in ("Glasses", "Cigarette"):
            return True
        px1, py1, px2, py2 = p_box
        ph = max(py2 - py1, 1e-6)
        pw = max(px2 - px1, 1e-6)
        icy = (item_box[1] + item_box[3]) / 2
        rel_y = (icy - py1) / ph  # 0 = testa, 1 = piedi
        if cat == "Glasses":
            return rel_y <= self._GLASSES_MAX_REL_Y
        # Cigarette
        iw = max(item_box[2] - item_box[0], 0)
        ih = max(item_box[3] - item_box[1], 0)
        area_ratio = (iw * ih) / (pw * ph)
        width_ratio = iw / pw
        return (
            area_ratio <= self._CIGARETTE_MAX_AREA_RATIO
            and width_ratio <= self._CIGARETTE_MAX_WIDTH_RATIO
            and rel_y <= self._CIGARETTE_MAX_REL_Y
        )

    def _expand_bbox(self, p_box: tuple, fw: int, fh: int, ppe_cat: str) -> tuple:
        """
        Ritorna una bbox allargata della persona per catturare
        oggetti che fisicamente escono dal bordo del corpo.
        - Glove/Cigarette: +40% larghezza su entrambi i lati
        - Shoe:  +40% altezza verso il basso, +15% larghezza
        - Helmet: +20% altezza verso l'alto, +10% larghezza
        """
        x1, y1, x2, y2 = p_box
        w = x2 - x1
        h = y2 - y1
        if ppe_cat in ("Glove", "Cigarette"):
            pad_x = w * 0.40
            return (max(0, x1 - pad_x), y1, min(fw, x2 + pad_x), y2)
        if ppe_cat == "Shoe":
            pad_y = h * 0.40
            pad_x = w * 0.15
            return (max(0, x1 - pad_x), y1, min(fw, x2 + pad_x), min(fh, y2 + pad_y))
        if ppe_cat == "Helmet":
            pad_y = h * 0.20
            pad_x = w * 0.10
            return (max(0, x1 - pad_x), max(0, y1 - pad_y), min(fw, x2 + pad_x), y2)
        return p_box

    def check(self, detections: list, frame_w: int = 640, frame_h: int = 640) -> list[PersonPPEResult]:
        """
        Ritorna un PersonPPEResult per ogni Person rilevata.
        frame_w/frame_h servono per denormalizzare bbox se necessario.
        """
        persons = [(i, d) for i, d in enumerate(detections) if d.label.lower() == "person"]
        # Ogni item porta con se' un flag `prohibited` per instradarlo nel
        # ramo giusto dopo l'associazione (stessa geometria per entrambi).
        items = [
            (d, _PPE_LABEL_MAP[d.label.lower().strip()], False)
            for d in detections
            if d.label.lower().strip() in _PPE_LABEL_MAP
        ] + [
            (d, _PROHIBITED_LABEL_MAP[d.label.lower().strip()], True)
            for d in detections
            if d.label.lower().strip() in _PROHIBITED_LABEL_MAP
        ]

        if not persons:
            return []

        # Pre-calcola le bbox di tutte le persone
        person_boxes = {i: _to_xyxy(d.bbox, frame_w, frame_h) for i, d in persons}

        # ── Greedy assignment: ogni item va alla persona con overlap maggiore ──
        # Evita che uno stesso item venga contato per più persone
        assignments: dict[int, list[tuple[str, object, bool]]] = {i: [] for i, _ in persons}

        for item_det, cat, prohibited in items:
            item_box = _to_xyxy(item_det.bbox, frame_w, frame_h)
            if item_box is None:
                continue

            best_score = -1.0
            best_pidx = None
            for p_i, _ in persons:
                p_box = person_boxes.get(p_i)
                if p_box is None:
                    continue
                # Usa la bbox espansa per gli oggetti che escono dal corpo
                search_box = self._expand_bbox(p_box, frame_w, frame_h, cat) if cat in self._EXPANDED_PPE else p_box
                score = _overlap_ratio(item_box, search_box)
                if score >= self.containment_threshold and score > best_score:
                    best_score = score
                    best_pidx = p_i

            if best_pidx is not None and self._plausible(cat, item_box, person_boxes[best_pidx]):
                assignments[best_pidx].append((cat, item_det, prohibited))

        # ── Costruisci i risultati ─────────────────────────────────────────────
        results: list[PersonPPEResult] = []
        for p_i, p_det in persons:
            p_box = person_boxes[p_i]

            found_counts: dict[str, int] = dict.fromkeys(self.required_ppe, 0)
            prohibited_found: list[str] = []
            assoc_dets = []
            for cat, det, prohibited in assignments[p_i]:
                assoc_dets.append(det)
                if prohibited:
                    if cat not in prohibited_found:
                        prohibited_found.append(cat)
                else:
                    found_counts[cat] = found_counts.get(cat, 0) + 1

            if p_box is None:
                missing = list(self.required_ppe.keys()) if self.no_bbox_policy == "all_missing" else []
            else:
                missing = [cat for cat, req in self.required_ppe.items() if found_counts.get(cat, 0) < req]

            results.append(
                PersonPPEResult(
                    person_idx=p_i,
                    person_bbox=p_box,
                    person_conf=p_det.confidence,
                    frame_size=(frame_w, frame_h),
                    found_ppe=found_counts,
                    missing_ppe=missing,
                    prohibited_present=sorted(prohibited_found),
                    associated_ppe=assoc_dets,
                )
            )

        return results
