"""
Real-Time Video Safety Tracker.

Architecture (hybrid cascade):

  ┌──────────────┐     every frame      ┌──────────────────────┐
  │  YOLO (fast) │ ──── detections ───► │  PersonPPEChecker    │
  └──────────────┘                      │  (IoU association)   │
                                        └──────────┬───────────┘
                                                   │ non-compliant persons
                                        ┌──────────▼───────────┐
                                        │  PersistenceTracker  │
                                        │  (N consecutive      │
                                        │   frames threshold)  │
                                        └──────────┬───────────┘
                                                   │ confirmed violations
                                        ┌──────────▼───────────┐
                                        │  VLM Validator       │  (local,
                                        │  (Florence-2 or      │   optional)
                                        │   Moondream2)        │
                                        └──────────┬───────────┘
                                                   │ validated alerts
                                        ┌──────────▼───────────┐
                                        │  Annotated output    │
                                        │  (cv2 display / file)│
                                        └──────────────────────┘

Usage:
    tracker = VideoSafetyTracker(
        yolo_analyzer=YOLOAnalyzer("weights/best.onnx"),
        vlm_validator=Florence2Analyzer(),   # optional
        persistence_frames=8,
    )
    tracker.run("site_camera.mp4")           # file
    tracker.run(0)                           # webcam
    tracker.run("rtsp://...")                # IP camera
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2 as cv

from .analyzer import (
    BaseAnalyzer,
    YOLOAnalyzer,
)
from .person_ppe_checker import PersonPPEChecker, PersonPPEResult

if TYPE_CHECKING:
    import numpy as np

# ── Alert record ──────────────────────────────────────────────────────────────


@dataclass
class FrameAlert:
    frame_idx: int
    timestamp_s: float
    person_results: list[PersonPPEResult]
    vlm_confirmed: bool = False
    vlm_response: str | None = None

    @property
    def violations(self) -> list[PersonPPEResult]:
        return [p for p in self.person_results if not p.is_compliant]

    def summary(self) -> str:
        viol = self.violations
        suffix = " [VLM✓]" if self.vlm_confirmed else ""
        return (
            f"[t={self.timestamp_s:.1f}s frame={self.frame_idx}] "
            f"{len(viol)} person(s) in violation{suffix}: " + "; ".join(p.summary() for p in viol)
        )


# ── Persistence tracker (per-person, based on missing PPE set) ────────────────
class VideoViolationTracker:
    """
    Tracks consecutive frames where a *specific* PPE violation persists.
    Uses a simple sliding window keyed by (approximate_bbox_region, missing_set).

    Strategy: we don't require exact person re-identification — instead we
    discretise the frame into grid cells and consider a violation "the same"
    if it appears in the same cell with the same missing items.
    """

    def __init__(self, threshold_frames: int = 8, grid: int = 8):
        self.threshold = threshold_frames
        self.grid = grid
        # key → deque of booleans (True = violation present in that frame)
        self._history: dict[str, deque] = {}

    def _key(self, pr: PersonPPEResult, frame_w: int, frame_h: int) -> str:
        if pr.person_bbox:
            cx = int(((pr.person_bbox[0] + pr.person_bbox[2]) / 2) / max(frame_w, 1) * self.grid)
            cy = int(((pr.person_bbox[1] + pr.person_bbox[3]) / 2) / max(frame_h, 1) * self.grid)
        else:
            cx, cy = -1, -1
        missing_key = ",".join(sorted(pr.missing_ppe))
        return f"{cx}:{cy}:{missing_key}"

    def update(
        self,
        person_results: list[PersonPPEResult],
        frame_w: int,
        frame_h: int,
    ) -> list[PersonPPEResult]:
        """
        Return list of PersonPPEResult whose violation has persisted for
        >= threshold consecutive frames.
        """
        active_keys: set[str] = set()

        for pr in person_results:
            if pr.is_compliant:
                continue
            k = self._key(pr, frame_w, frame_h)
            active_keys.add(k)
            if k not in self._history:
                self._history[k] = deque(maxlen=self.threshold)
            self._history[k].append(True)

        # Mark absent violations as False
        for k in list(self._history.keys()):
            if k not in active_keys:
                self._history[k].append(False)
                if not any(self._history[k]):
                    del self._history[k]

        # Confirmed = threshold consecutive Trues
        confirmed_keys = {k for k, dq in self._history.items() if len(dq) == self.threshold and all(dq)}

        return [pr for pr in person_results if not pr.is_compliant and self._key(pr, frame_w, frame_h) in confirmed_keys]


# ── VLM validation prompt builder ─────────────────────────────────────────────
def _build_vlm_prompt(missing_ppe: list[str]) -> str:
    items = ", ".join(missing_ppe)
    return (
        f"This is a cropped image of a construction worker. "
        f"YOLO detected that the following PPE items may be MISSING: {items}. "
        f"Please verify. For each item in the list answer YES (missing) or NO (present). "
        f"Then respond ONLY with a JSON object like: "
        f'{{"Helmet": true, "Vest": false, "Glove": true, "Shoe": false}} '
        f"where true means MISSING, false means PRESENT."
    )


# ── Drawing helpers ───────────────────────────────────────────────────────────
_COLORS = {
    "COMPLIANT": (40, 200, 60),  # green
    "VIOLATION": (0, 60, 230),  # red
    "PENDING": (20, 180, 230),  # yellow
    "PPE_OK": (40, 200, 60),
    "PPE_MISSING": (0, 60, 230),
}


def _draw_person(
    img: np.ndarray,
    pr: PersonPPEResult,
    confirmed: bool,
    frame_idx: int,
) -> None:
    if pr.person_bbox is None:
        return

    x1, y1, x2, y2 = (int(v) for v in pr.person_bbox)

    if pr.is_compliant:
        color = _COLORS["COMPLIANT"]
        label = "PPE OK"
    elif confirmed:
        color = _COLORS["VIOLATION"]
        label = f"ALERT: {','.join(pr.missing_ppe)}"
    else:
        color = _COLORS["PENDING"]
        label = f"Pending: {','.join(pr.missing_ppe)}"

    cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv.putText(img, label, (x1, max(y1 - 6, 14)), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv.LINE_AA)

    # PPE checklist mini-overlay
    checklist_y = y1 + 18
    for cat, required in PersonPPEChecker.FULL_PPE.items():
        found = pr.found_ppe.get(cat, 0)
        ok = found >= required
        icon = "✔" if ok else "✘"
        c = _COLORS["PPE_OK"] if ok else _COLORS["PPE_MISSING"]
        cv.putText(img, f"{icon}{cat}", (x1 + 4, checklist_y), cv.FONT_HERSHEY_SIMPLEX, 0.40, c, 1, cv.LINE_AA)
        checklist_y += 16


def _draw_hud(
    img: np.ndarray,
    frame_idx: int,
    fps: float,
    yolo_ms: float,
    n_persons: int,
    n_alerts: int,
    vlm_calls: int,
) -> None:
    _h, _w = img.shape[:2]
    overlay = img.copy()
    cv.rectangle(overlay, (0, 0), (320, 120), (30, 30, 30), -1)
    cv.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    lines = [
        f"Frame: {frame_idx}  FPS: {fps:.1f}",
        f"YOLO: {yolo_ms:.1f}ms",
        f"Persons: {n_persons}  Alerts: {n_alerts}",
        f"VLM calls: {vlm_calls}",
    ]
    for i, line in enumerate(lines):
        cv.putText(img, line, (8, 22 + i * 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv.LINE_AA)


# ── Main tracker class ────────────────────────────────────────────────────────
class VideoSafetyTracker:
    """
    Full real-time video safety tracking pipeline.

    Parameters
    ----------
    yolo_analyzer       : YOLOAnalyzer instance (must be pre-configured)
    vlm_validator       : Optional local VLM (Florence2Analyzer / MoondreamAnalyzer)
    ppe_checker         : PersonPPEChecker (default full PPE set)
    persistence_frames  : Consecutive frames before triggering VLM/alert (default 8)
    skip_frames         : Run YOLO every N frames for speed (default 1 = every frame)
    display             : Show live OpenCV window
    save_output         : Path to save annotated video (None = don't save)
    alert_log_path      : Path to save alert log JSON (None = don't save)
    max_alerts          : Stop after this many confirmed alerts (0 = unlimited)
    """

    def __init__(
        self,
        yolo_analyzer: YOLOAnalyzer,
        vlm_validator: BaseAnalyzer | None = None,
        ppe_checker: PersonPPEChecker | None = None,
        persistence_frames: int = 8,
        skip_frames: int = 1,
        display: bool = True,
        save_output: str | Path | None = None,
        alert_log_path: str | Path | None = None,
        max_alerts: int = 0,
    ):
        self.yolo = yolo_analyzer
        self.vlm = vlm_validator
        self.checker = ppe_checker or PersonPPEChecker()
        self.persistence_frames = persistence_frames
        self.skip_frames = max(1, skip_frames)
        self.display = display
        self.save_output = Path(save_output) if save_output else None
        self.alert_log_path = Path(alert_log_path) if alert_log_path else None
        self.max_alerts = max_alerts

        self._vio_tracker = VideoViolationTracker(threshold_frames=persistence_frames)
        self._vlm_calls = 0
        self._alerts: list[FrameAlert] = []

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, source: str | int = 0) -> list[FrameAlert]:
        """
        Run the pipeline on a video source.

        source: path to video file, RTSP URL, or webcam index (int).
        Returns list of confirmed FrameAlert objects.
        """
        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            msg = f"Cannot open video source: {source}"
            raise RuntimeError(msg)

        fps_src = cap.get(cv.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.save_output:
            self.save_output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            writer = cv.VideoWriter(str(self.save_output), fourcc, fps_src, (frame_w, frame_h))

        frame_idx = 0
        last_person_results: list[PersonPPEResult] = []
        last_yolo_ms = 0.0
        fps_counter = deque(maxlen=30)
        t_last = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                t_now = time.perf_counter()
                fps_counter.append(1.0 / max(t_now - t_last, 1e-6))
                t_last = t_now
                live_fps = sum(fps_counter) / len(fps_counter)

                # ── Step 1: YOLO inference (every skip_frames) ───────────────
                if frame_idx % self.skip_frames == 1 or self.skip_frames == 1:
                    tmp_path = "/tmp/_tracker_frame.jpg"
                    cv.imwrite(tmp_path, frame)
                    yolo_result = self.yolo.analyze(tmp_path)
                    last_yolo_ms = yolo_result.inference_time_ms

                    # ── Step 2: Per-person PPE check ─────────────────────────
                    last_person_results = self.checker.check(yolo_result.detections)

                # ── Step 3: Persistence filtering ────────────────────────────
                confirmed_violations = self._vio_tracker.update(last_person_results, frame_w, frame_h)

                # ── Step 4: VLM validation for confirmed violations ───────────
                if confirmed_violations:
                    alert = self._maybe_validate_vlm(
                        frame,
                        frame_idx,
                        cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0,
                        last_person_results,
                        confirmed_violations,
                    )
                    if alert:
                        self._alerts.append(alert)
                        print(alert.summary())
                        if self.max_alerts and len(self._alerts) >= self.max_alerts:
                            break

                # ── Step 5: Annotate frame ────────────────────────────────────
                confirmed_set = {id(p) for p in confirmed_violations}
                for pr in last_person_results:
                    _draw_person(frame, pr, id(pr) in confirmed_set, frame_idx)

                _draw_hud(
                    frame,
                    frame_idx,
                    live_fps,
                    last_yolo_ms,
                    len(last_person_results),
                    len(confirmed_violations),
                    self._vlm_calls,
                )

                if writer:
                    writer.write(frame)

                if self.display:
                    cv.imshow("PPE Safety Tracker", frame)
                    if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

        finally:
            cap.release()
            if writer:
                writer.release()
            if self.display:
                cv.destroyAllWindows()
            if self.alert_log_path:
                self._save_alert_log()

        return self._alerts

    # ── VLM helper ────────────────────────────────────────────────────────────
    def _maybe_validate_vlm(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        all_person_results: list[PersonPPEResult],
        confirmed: list[PersonPPEResult],
    ) -> FrameAlert | None:
        """
        For each confirmed violation, crop the person ROI and send to VLM.
        Returns a FrameAlert (possibly with vlm_confirmed=True).
        """
        alert = FrameAlert(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            person_results=all_person_results,
        )

        if self.vlm is None:
            alert.vlm_confirmed = True  # trust YOLO + tracker
            return alert

        # Crop + save temp file per violation person
        any_confirmed_by_vlm = False
        for pr in confirmed:
            if not pr.needs_vlm_validation:
                alert.vlm_confirmed = True
                any_confirmed_by_vlm = True
                continue

            crop = self._crop_person(frame, pr)
            tmp_crop = "/tmp/_tracker_crop.jpg"
            cv.imwrite(tmp_crop, crop)

            vlm_result = self.vlm.analyze(tmp_crop)
            self._vlm_calls += 1

            # Simple heuristic: if VLM finds any violation label → confirmed
            vlm_violation_labels = [d.label for d in vlm_result.detections if d.is_violation]
            if vlm_violation_labels or vlm_result.error is None:
                any_confirmed_by_vlm = True
                alert.vlm_response = str([d.label for d in vlm_result.detections])

        alert.vlm_confirmed = any_confirmed_by_vlm
        return alert

    @staticmethod
    def _crop_person(frame: np.ndarray, pr: PersonPPEResult) -> np.ndarray:
        if pr.person_bbox is None:
            return frame
        h, w = frame.shape[:2]
        x1 = max(0, int(pr.person_bbox[0]))
        y1 = max(0, int(pr.person_bbox[1]))
        x2 = min(w, int(pr.person_bbox[2]))
        y2 = min(h, int(pr.person_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]

    # ── Alert log ─────────────────────────────────────────────────────────────
    def _save_alert_log(self) -> None:
        import json

        data = []
        for a in self._alerts:
            data.append(
                {
                    "frame": a.frame_idx,
                    "timestamp_s": round(a.timestamp_s, 2),
                    "vlm_confirmed": a.vlm_confirmed,
                    "vlm_response": a.vlm_response,
                    "violations": [
                        {
                            "person_idx": p.person_idx,
                            "missing_ppe": p.missing_ppe,
                            "found_ppe": p.found_ppe,
                            "bbox": list(p.person_bbox) if p.person_bbox else None,
                        }
                        for p in a.violations
                    ],
                }
            )
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_log_path.write_text(json.dumps(data, indent=2))
        print(f"Alert log saved → {self.alert_log_path}")


# ── Convenience factory ───────────────────────────────────────────────────────
def build_hybrid_tracker(
    yolo_model_path: str,
    vlm_backend: str = "florence2",  # "florence2" | "moondream" | "none"
    vlm_device: str = "cpu",
    persistence_frames: int = 8,
    skip_frames: int = 1,
    display: bool = True,
    save_output: str | None = None,
    alert_log: str | None = None,
    yolo_conf: float = 0.40,
) -> VideoSafetyTracker:
    """
    Quick-start factory.  Instantiates YOLO + optional local VLM and returns
    a ready-to-use VideoSafetyTracker.

    Example:
        tracker = build_hybrid_tracker(
            yolo_model_path="weights/best.onnx",
            vlm_backend="florence2",
            save_output="output/annotated.mp4",
            alert_log="output/alerts.json",
        )
        tracker.run("rtsp://camera1")
    """
    yolo = YOLOAnalyzer(model_path=yolo_model_path, conf_threshold=yolo_conf)

    vlm: BaseAnalyzer | None = None
    if vlm_backend == "florence2":
        from .analyzer import Florence2Analyzer

        vlm = Florence2Analyzer(device=vlm_device)
    elif vlm_backend == "moondream":
        from .analyzer import MoondreamAnalyzer

        vlm = MoondreamAnalyzer(device=vlm_device)
    elif vlm_backend == "none":
        vlm = None
    else:
        msg = f"Unknown vlm_backend: {vlm_backend!r}. Choose 'florence2', 'moondream', or 'none'."
        raise ValueError(msg)

    return VideoSafetyTracker(
        yolo_analyzer=yolo,
        vlm_validator=vlm,
        persistence_frames=persistence_frames,
        skip_frames=skip_frames,
        display=display,
        save_output=save_output,
        alert_log_path=alert_log,
    )
