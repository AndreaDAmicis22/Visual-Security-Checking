"""
Person-centric PPE Completeness Checker.

For every 'Person' detected by YOLO, checks whether required PPE items
(Helmet, Vest, Glove ×2, Shoe ×2) are spatially associated via containment
overlap with that person's bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Required PPE ──────────────────────────────────────────────────────────────

REQUIRED_PPE_COUNTS: dict[str, int] = {
    "Helmet": 1,
    "Vest": 1,
    "Glove": 2,
    "Shoe": 2,
}

# All lowercase label strings → canonical PPE category
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


# ── Bbox helpers ──────────────────────────────────────────────────────────────
def _to_xyxy(bbox) -> tuple[float, float, float, float] | None:
    """
    Normalise any bbox format to (x1, y1, x2, y2).

    Accepted formats:
      [x1, y1, x2, y2]               flat 4-element list/tuple
      [[x1,y1], [x2,y2], ...]        list of (x,y) points (polygon)
    Returns None if the input cannot be parsed or is degenerate.
    """
    if bbox is None:
        return None
    try:
        # Detect whether elements are scalars or 2-element sequences
        first = bbox[0]
        if isinstance(first, (list, tuple)):
            # Polygon / point list → axis-aligned bbox
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            return (min(xs), min(ys), max(xs), max(ys))
        # Flat numeric list [x1, y1, x2, y2]
        coords = [float(v) for v in bbox]
        if len(coords) == 4:
            return (coords[0], coords[1], coords[2], coords[3])
        if len(coords) >= 8:
            xs = coords[0::2]
            ys = coords[1::2]
            return (min(xs), min(ys), max(xs), max(ys))
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    return None


def _containment(inner: tuple, outer: tuple) -> float:
    """Fraction of *inner* box area covered by *outer* (0.0–1.0)."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_inner = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area_inner if area_inner > 0 else 0.0


def _iou(a: tuple, b: tuple) -> float:
    """Standard IoU for (x1,y1,x2,y2) boxes."""
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


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class PersonPPEResult:
    person_idx: int
    person_bbox: tuple | None  # (x1,y1,x2,y2) in pixel coords
    person_conf: float
    found_ppe: dict[str, int] = field(default_factory=dict)
    missing_ppe: list[str] = field(default_factory=list)
    associated_ppe: list = field(default_factory=list)
    needs_vlm_validation: bool = False

    @property
    def is_compliant(self) -> bool:
        return len(self.missing_ppe) == 0

    def summary(self) -> str:
        status = "✅ OK" if self.is_compliant else f"⚠️  MISSING: {self.missing_ppe}"
        return f"Person#{self.person_idx}(conf={self.person_conf:.2f}) found={self.found_ppe} | {status}" + (
            " [VLM→]" if self.needs_vlm_validation else ""
        )


# ── Checker ───────────────────────────────────────────────────────────────────
class PersonPPEChecker:
    """
    Associates YOLO detections to individual persons and evaluates PPE
    completeness via spatial containment.

    Parameters
    ----------
    required_ppe:
        {category: min_count}.  Default = full PPE set.
    containment_threshold:
        Minimum fraction of a PPE item bbox that must lie *inside* the
        person bbox to count as associated.  Default 0.30 (relaxed to handle
        items at the edge of the person box, e.g. shoes near the bottom).
    iou_threshold:
        Secondary criterion used when both boxes are similarly sized.
        Default 0.05.
    no_bbox_policy:
        What to do when a person has no bbox ("none_missing" | "vlm_only").
        "none_missing" → treat as fully equipped (conservative).
        "vlm_only"     → mark as needing VLM with no PPE confirmed.
    """

    FULL_PPE = REQUIRED_PPE_COUNTS

    def __init__(
        self,
        required_ppe: dict[str, int] | None = None,
        containment_threshold: float = 0.30,
        iou_threshold: float = 0.05,
        no_bbox_policy: str = "vlm_only",
        vlm_trigger_missing: set[str] | None = None,
    ):
        self.required_ppe = required_ppe or dict(REQUIRED_PPE_COUNTS)
        self.containment_threshold = containment_threshold
        self.iou_threshold = iou_threshold
        self.no_bbox_policy = no_bbox_policy
        self.vlm_trigger_missing = vlm_trigger_missing  # None → trigger on any missing

    def check(self, detections: list) -> list[PersonPPEResult]:
        """
        Returns one PersonPPEResult per Person detection in `detections`.
        PPE items are greedily assigned to the *nearest* (highest-containment)
        person — each item can satisfy only one person.
        """
        persons = [(i, d) for i, d in enumerate(detections) if d.label.lower() == "person"]
        ppe_items = [
            (i, d, _PPE_LABEL_MAP[d.label.lower().strip()])
            for i, d in enumerate(detections)
            if d.label.lower().strip() in _PPE_LABEL_MAP
        ]

        # Pre-compute boxes
        person_boxes = {i: _to_xyxy(d.bbox) for i, d in persons}

        # ── Greedy assignment: assign each PPE item to best-matching person ──
        # score[p_idx][item_idx] = containment score
        assignments: dict[int, list[tuple[str, object]]] = {i: [] for i, _ in persons}

        for _item_i, item_det, ppe_cat in ppe_items:
            item_box = _to_xyxy(item_det.bbox)
            if item_box is None:
                continue

            best_score = -1.0
            best_pidx = None

            for p_i, _ in persons:
                p_box = person_boxes.get(p_i)
                if p_box is None:
                    continue
                score = _containment(item_box, p_box)
                if score < self.containment_threshold:
                    # fallback: try IoU for same-scale objects (e.g. vest ≈ torso)
                    score = _iou(item_box, p_box)
                    if score < self.iou_threshold:
                        continue
                if score > best_score:
                    best_score = score
                    best_pidx = p_i

            if best_pidx is not None:
                assignments[best_pidx].append((ppe_cat, item_det))

        # ── Build results ────────────────────────────────────────────────────
        results: list[PersonPPEResult] = []

        for p_i, p_det in persons:
            p_box = person_boxes[p_i]

            # Count per-category
            found_counts: dict[str, int] = dict.fromkeys(self.required_ppe, 0)
            assoc_dets = []
            for ppe_cat, det in assignments[p_i]:
                found_counts[ppe_cat] = found_counts.get(ppe_cat, 0) + 1
                assoc_dets.append(det)

            # Handle no-bbox persons
            if p_box is None:
                if self.no_bbox_policy == "vlm_only":
                    # We cannot spatially verify → send to VLM, assume all missing
                    missing = list(self.required_ppe.keys())
                    needs_vlm = True
                else:
                    missing = []
                    needs_vlm = False
            else:
                missing = [cat for cat, req_n in self.required_ppe.items() if found_counts.get(cat, 0) < req_n]
                if missing:
                    needs_vlm = True if self.vlm_trigger_missing is None else bool(set(missing) & self.vlm_trigger_missing)
                else:
                    needs_vlm = False

            results.append(
                PersonPPEResult(
                    person_idx=p_i,
                    person_bbox=p_box,
                    person_conf=p_det.confidence,
                    found_ppe=found_counts,
                    missing_ppe=missing,
                    associated_ppe=assoc_dets,
                    needs_vlm_validation=needs_vlm,
                )
            )

        return results
