"""
Person-centric PPE Completeness Checker.

For every 'Person' detected by YOLO, this module checks whether the required
PPE items (Helmet, Vest, Glove ×2, Shoe ×2) are spatially associated with
that person's bounding box via IoU / containment overlap.

If one or more items are missing, the person is flagged as a violation and
(optionally) routed to the VLM validator for confirmation.

Usage:
    checker = PersonPPEChecker(required_ppe=PersonPPEChecker.FULL_PPE)
    results = checker.check(yolo_detections)
    for pr in results:
        print(pr.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import Detection

# ── Required PPE set ──────────────────────────────────────────────────────────
# Counts: how many instances of each item are required per person
REQUIRED_PPE_COUNTS: dict[str, int] = {
    "Helmet": 1,
    "Vest": 1,
    "Glove": 2,
    "Shoe": 2,
}

# Labels produced by YOLO that map directly to a PPE category
# (lowercase normalised)
_PPE_LABEL_MAP: dict[str, str] = {
    "helmet": "Helmet",
    "vest": "Vest",
    "glove": "Glove",
    "shoe": "Shoe",
    # Common alternative spellings / dataset variants
    "hardhat": "Helmet",
    "hard-hat": "Helmet",
    "hard_hat": "Helmet",
    "hi-vis": "Vest",
    "hiviz": "Vest",
    "boot": "Shoe",
    "boots": "Shoe",
    "safety-boot": "Shoe",
}


def _bbox_to_xyxy(bbox) -> tuple[float, float, float, float] | None:
    """
    Accepts bbox in several formats:
      - [x1, y1, x2, y2]  (4-element flat list)
      - [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  (polygon)
    Returns (x1, y1, x2, y2) or None if not parseable.
    """
    if bbox is None:
        return None
    try:
        flat = [
            float(v)
            for pt in (bbox if isinstance(bbox[0], (list, tuple)) else [bbox])
            for v in (pt if isinstance(pt, (list, tuple)) else [pt])
        ]
        if len(flat) == 4:
            return (flat[0], flat[1], flat[2], flat[3])
        if len(flat) >= 8:  # polygon — use axis-aligned bounding box
            xs = flat[0::2]
            ys = flat[1::2]
            return (min(xs), min(ys), max(xs), max(ys))
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
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


def _containment(inner: tuple, outer: tuple) -> float:
    """
    Fraction of *inner* box area that is inside *outer*.
    Used to associate small PPE items with a large Person box.
    """
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_inner = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area_inner if area_inner > 0 else 0.0


# ── Per-person result ─────────────────────────────────────────────────────────
@dataclass
class PersonPPEResult:
    person_idx: int  # index in original detections list
    person_bbox: tuple | None  # (x1,y1,x2,y2) or None
    person_conf: float
    found_ppe: dict[str, int] = field(default_factory=dict)  # {category: count found}
    missing_ppe: list[str] = field(default_factory=list)  # e.g. ["Glove", "Shoe"]
    associated_ppe: list[Detection] = field(default_factory=list)
    needs_vlm_validation: bool = False

    @property
    def is_compliant(self) -> bool:
        return len(self.missing_ppe) == 0

    def summary(self) -> str:
        status = "✅ COMPLIANT" if self.is_compliant else f"⚠️  VIOLATION — missing: {self.missing_ppe}"
        return f"Person #{self.person_idx} (conf={self.person_conf:.2f}) | found PPE: {self.found_ppe} | {status}" + (
            " [→ VLM check scheduled]" if self.needs_vlm_validation else ""
        )


# ── Checker ───────────────────────────────────────────────────────────────────
class PersonPPEChecker:
    """
    Associates YOLO detections to individual persons and evaluates PPE completeness.

    Parameters
    ----------
    required_ppe:
        Dict mapping PPE category → minimum count per person.
        Default = REQUIRED_PPE_COUNTS (Helmet:1, Vest:1, Glove:2, Shoe:2).
    containment_threshold:
        Minimum fraction of a PPE item's bbox that must lie within the person
        bbox to be considered associated. Default 0.4.
    iou_threshold:
        Fallback IoU threshold used when both boxes are large.  Default 0.1.
    vlm_trigger_missing:
        If a person is missing *any* PPE in this set, schedule VLM validation.
        None → always schedule when non-compliant.
    """

    FULL_PPE = REQUIRED_PPE_COUNTS

    def __init__(
        self,
        required_ppe: dict[str, int] | None = None,
        containment_threshold: float = 0.40,
        iou_threshold: float = 0.10,
        vlm_trigger_missing: set[str] | None = None,
    ):
        self.required_ppe = required_ppe or REQUIRED_PPE_COUNTS
        self.containment_threshold = containment_threshold
        self.iou_threshold = iou_threshold
        self.vlm_trigger_missing = vlm_trigger_missing  # None → trigger on any missing

    # ── Public API ────────────────────────────────────────────────────────────
    def check(self, detections: list[Detection]) -> list[PersonPPEResult]:
        """
        Given a flat list of Detection objects (YOLO output for one frame),
        return one PersonPPEResult per detected person.
        """
        persons = [(i, d) for i, d in enumerate(detections) if d.label.lower() == "person"]
        ppe_items = [
            (i, d, self._normalise_label(d.label))
            for i, d in enumerate(detections)
            if self._normalise_label(d.label) in self.required_ppe
        ]

        results: list[PersonPPEResult] = []

        for p_idx, p_det in persons:
            p_box = _bbox_to_xyxy(p_det.bbox)
            found_counts: dict[str, int] = dict.fromkeys(self.required_ppe, 0)
            associated: list[Detection] = []

            for _, ppe_det, ppe_cat in ppe_items:
                ppe_box = _bbox_to_xyxy(ppe_det.bbox)
                if ppe_box is None or p_box is None:
                    # No spatial info — can't associate; fall through to VLM
                    continue

                cont = _containment(ppe_box, p_box)
                iou = _iou(ppe_box, p_box)

                if cont >= self.containment_threshold or iou >= self.iou_threshold:
                    found_counts[ppe_cat] = found_counts.get(ppe_cat, 0) + 1
                    associated.append(ppe_det)

            missing = [cat for cat, required_n in self.required_ppe.items() if found_counts.get(cat, 0) < required_n]

            # Decide VLM trigger
            if missing:
                needs_vlm = True if self.vlm_trigger_missing is None else bool(set(missing) & self.vlm_trigger_missing)
            else:
                needs_vlm = False

            # If person bbox is None (partial crop), always escalate to VLM
            if p_box is None and missing:
                needs_vlm = True

            results.append(
                PersonPPEResult(
                    person_idx=p_idx,
                    person_bbox=p_box,
                    person_conf=p_det.confidence,
                    found_ppe=found_counts,
                    missing_ppe=missing,
                    associated_ppe=associated,
                    needs_vlm_validation=needs_vlm,
                )
            )

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _normalise_label(label: str) -> str | None:
        return _PPE_LABEL_MAP.get(label.lower().strip())
