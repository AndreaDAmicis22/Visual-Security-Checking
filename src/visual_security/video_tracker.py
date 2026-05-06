"""
Real-Time Video Safety Tracker  — hybrid cascade pipeline.

Architecture:

  ┌─────────────┐  every N frames   ┌──────────────────────┐
  │  YOLO(fast) │ ── detections ──► │  PersonPPEChecker    │
  └─────────────┘                   │  (IoU containment)   │
                                    └──────────┬───────────┘
                                               │ PersonPPEResult list
                                    ┌──────────▼───────────┐
                                    │  VideoViolationTracker│
                                    │  (sliding-window per  │
                                    │   grid-cell+missing)  │
                                    └──────────┬───────────┘
                                               │ confirmed violations
                                    ┌──────────▼───────────┐
                                    │  VLM Validator        │  (local,
                                    │  Florence-2 /         │   optional)
                                    │  Moondream2           │
                                    └──────────┬───────────┘
                                               │ FrameAlert
                                    ┌──────────▼───────────┐
                                    │  Annotated output     │
                                    │  (cv2 window / file)  │
                                    └──────────────────────┘
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2 as cv

from .person_ppe_checker import PersonPPEChecker, PersonPPEResult

if TYPE_CHECKING:
    import numpy as np

    from .analyzer import BaseAnalyzer, YOLOAnalyzer

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
        suffix = " [VLM✓]" if self.vlm_confirmed else " [YOLO+tracker]"
        parts = "; ".join(p.summary() for p in viol)
        return f"[t={self.timestamp_s:.1f}s  frame={self.frame_idx}]  {len(viol)} violation(s){suffix}: {parts}"


# ── Persistence / violation tracker ──────────────────────────────────────────


class VideoViolationTracker:
    """
    Sliding-window tracker keyed by (grid_cell, frozenset_of_missing_ppe).

    A violation is "confirmed" once it has appeared in ≥ threshold of the
    last `window` frames.  This is more robust than requiring *consecutive*
    frames (a single missed YOLO detection won't reset the counter).

    Parameters
    ----------
    threshold_frames : int
        Minimum number of positive frames inside the window. Default 5.
    window : int
        Sliding-window width in frames. Default 8.
        Set window == threshold_frames for strict consecutive behaviour.
    grid : int
        Frame is divided into grid×grid cells for person tracking. Default 6.
    cooldown_frames : int
        After an alert fires, suppress new alerts for this many frames
        for the same key.  Prevents alert spam.  Default 30.
    """

    def __init__(
        self,
        threshold_frames: int = 5,
        window: int = 8,
        grid: int = 6,
        cooldown_frames: int = 30,
    ):
        self.threshold = threshold_frames
        self.window = window
        self.grid = grid
        self.cooldown = cooldown_frames

        # key → deque of booleans (True = violation seen in that frame)
        self._history: dict[str, deque] = {}
        # key → frames until cooldown expires
        self._cooldown: dict[str, int] = {}

    # ── Private ───────────────────────────────────────────────────────────────

    def _key(self, pr: PersonPPEResult, frame_w: int, frame_h: int) -> str:
        if pr.person_bbox:
            cx = int(((pr.person_bbox[0] + pr.person_bbox[2]) / 2) / max(frame_w, 1) * self.grid)
            cy = int(((pr.person_bbox[1] + pr.person_bbox[3]) / 2) / max(frame_h, 1) * self.grid)
            cx = min(cx, self.grid - 1)
            cy = min(cy, self.grid - 1)
        else:
            cx, cy = -1, -1
        missing_key = "|".join(sorted(pr.missing_ppe))
        return f"{cx}:{cy}:{missing_key}"

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        person_results: list[PersonPPEResult],
        frame_w: int,
        frame_h: int,
    ) -> list[PersonPPEResult]:
        """
        Call once per processed frame.
        Returns the subset of non-compliant persons whose violation is confirmed.
        """
        active_keys: set[str] = set()

        # 1. Register violations seen this frame
        for pr in person_results:
            if pr.is_compliant:
                continue
            k = self._key(pr, frame_w, frame_h)
            active_keys.add(k)
            if k not in self._history:
                self._history[k] = deque(maxlen=self.window)
            self._history[k].append(True)

        # 2. For every tracked key NOT seen this frame → push False
        for k in list(self._history.keys()):
            if k not in active_keys:
                self._history[k].append(False)
                # Prune keys that have no recent positive frames
                if not any(self._history[k]):
                    del self._history[k]
                    self._cooldown.pop(k, None)

        # 3. Tick down cooldowns
        for k in list(self._cooldown.keys()):
            self._cooldown[k] -= 1
            if self._cooldown[k] <= 0:
                del self._cooldown[k]

        # 4. A key is "confirmed" when it has enough positives in the window
        #    AND is not in cooldown
        confirmed_keys: set[str] = set()
        for k, dq in self._history.items():
            if k in self._cooldown:
                continue
            if sum(dq) >= self.threshold:
                confirmed_keys.add(k)
                self._cooldown[k] = self.cooldown  # arm cooldown

        # 5. Map back to PersonPPEResult objects
        return [pr for pr in person_results if not pr.is_compliant and self._key(pr, frame_w, frame_h) in confirmed_keys]

    def person_state(
        self,
        pr: PersonPPEResult,
        frame_w: int,
        frame_h: int,
    ) -> float:
        """Return fraction of window frames this person's violation was seen (0–1)."""
        k = self._key(pr, frame_w, frame_h)
        dq = self._history.get(k)
        if not dq:
            return 0.0
        return sum(dq) / len(dq)


# ── Drawing helpers ───────────────────────────────────────────────────────────

# BGR colours
_C_GREEN = (40, 200, 60)
_C_RED = (0, 50, 220)
_C_YELLOW = (20, 190, 230)
_C_GREY = (160, 160, 160)
_C_WHITE = (230, 230, 230)


def _draw_person(
    img: np.ndarray,
    pr: PersonPPEResult,
    confirmed: bool,
    fill: float,  # 0–1, fraction of window filled
) -> None:
    """Draw bounding box + PPE checklist on the frame for one person."""
    if pr.person_bbox is None:
        return

    x1, y1, x2, y2 = (int(v) for v in pr.person_bbox)

    if pr.is_compliant:
        color = _C_GREEN
        label = "PPE OK"
        thickness = 2
    elif confirmed:
        color = _C_RED
        label = f"ALERT: {', '.join(pr.missing_ppe)}"
        thickness = 3
    else:
        # pending — colour intensity grows with fill ratio
        b = int(20 + fill * 210)
        g = int(190 - fill * 130)
        r = int(230 - fill * 170)
        color = (b, g, r)
        label = f"[{int(fill * 100):2d}%] {', '.join(pr.missing_ppe)}"
        thickness = 2

    cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv.putText(img, label, (x1, max(y1 - 8, 14)), cv.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv.LINE_AA)

    # ── PPE checklist (bottom-left of box) ───────────────────────────────────
    cy = y1 + 18
    for cat, req in PersonPPEChecker.FULL_PPE.items():
        found = pr.found_ppe.get(cat, 0)
        ok = found >= req
        mark = chr(0x2714) if ok else chr(0x2718)  # ✔ / ✘
        c = _C_GREEN if ok else _C_RED
        cv.putText(img, f"{mark} {cat}({found}/{req})", (x1 + 4, cy), cv.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv.LINE_AA)
        cy += 15


def _draw_hud(
    img: np.ndarray,
    frame_idx: int,
    fps: float,
    yolo_ms: float,
    n_persons: int,
    n_pending: int,
    n_alerts: int,
    vlm_calls: int,
) -> None:
    """Semi-transparent HUD in the top-left corner."""
    overlay = img.copy()
    cv.rectangle(overlay, (0, 0), (330, 130), (20, 20, 20), -1)
    cv.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    lines = [
        f"Frame {frame_idx:>6}   FPS {fps:>5.1f}",
        f"YOLO    {yolo_ms:>6.1f} ms",
        f"Persons {n_persons:>3}   Pending {n_pending:>3}",
        f"Alerts  {n_alerts:>3}   VLM calls {vlm_calls:>4}",
    ]
    for i, line in enumerate(lines):
        cv.putText(img, line, (8, 22 + i * 26), cv.FONT_HERSHEY_SIMPLEX, 0.52, _C_WHITE, 1, cv.LINE_AA)


# ── Main tracker ──────────────────────────────────────────────────────────────
class VideoSafetyTracker:
    """
    Full real-time video safety tracking pipeline.

    Parameters
    ----------
    yolo_analyzer       Pre-configured YOLOAnalyzer.
    vlm_validator       Optional local VLM (Florence2Analyzer / MoondreamAnalyzer).
    ppe_checker         PersonPPEChecker — defaults to full PPE set.
    persistence_frames  Positive frames needed inside the window to trigger.
    window_frames       Sliding-window width (≥ persistence_frames).
    skip_frames         Run YOLO every N frames (1 = every frame).
    display             Show live OpenCV window.
    save_output         Path for annotated output video.
    alert_log_path      Path for JSON alert log.
    max_alerts          Stop after N confirmed alerts (0 = unlimited).
    """

    def __init__(
        self,
        yolo_analyzer: YOLOAnalyzer,
        vlm_validator: BaseAnalyzer | None = None,
        ppe_checker: PersonPPEChecker | None = None,
        persistence_frames: int = 5,
        window_frames: int = 8,
        skip_frames: int = 1,
        display: bool = True,
        save_output: str | Path | None = None,
        alert_log_path: str | Path | None = None,
        max_alerts: int = 0,
    ):
        self.yolo = yolo_analyzer
        self.vlm = vlm_validator
        self.checker = ppe_checker or PersonPPEChecker()

        window = max(window_frames, persistence_frames)
        self._vio_tracker = VideoViolationTracker(
            threshold_frames=persistence_frames,
            window=window,
        )

        self.skip_frames = max(1, skip_frames)
        self.display = display
        self.save_output = Path(save_output) if save_output else None
        self.alert_log_path = Path(alert_log_path) if alert_log_path else None
        self.max_alerts = max_alerts

        self._vlm_calls = 0
        self._alerts: list[FrameAlert] = []

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, source: str | int = 0) -> list[FrameAlert]:
        """Run on a video file, RTSP stream, or webcam index."""
        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            msg = f"Cannot open video source: {source!r}"
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

                # ── Step 1: YOLO + PPE check (every skip_frames) ─────────────
                if (frame_idx - 1) % self.skip_frames == 0:
                    tmp_path = "/tmp/_tracker_frame.jpg"
                    cv.imwrite(tmp_path, frame)
                    yolo_result = self.yolo.analyze(tmp_path)
                    last_yolo_ms = yolo_result.inference_time_ms
                    last_person_results = self.checker.check(yolo_result.detections)

                # ── Step 2: Persistence filtering ────────────────────────────
                confirmed = self._vio_tracker.update(last_person_results, frame_w, frame_h)

                # ── Step 3: VLM validation & alert creation ───────────────────
                if confirmed:
                    alert = self._validate_and_alert(
                        frame,
                        frame_idx,
                        cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0,
                        last_person_results,
                        confirmed,
                    )
                    self._alerts.append(alert)
                    print(alert.summary())
                    if self.max_alerts and len(self._alerts) >= self.max_alerts:
                        break

                # ── Step 4: Annotate frame ────────────────────────────────────
                # Build a set of person indices that are confirmed (stable key)
                confirmed_idxs = {pr.person_idx for pr in confirmed}

                for pr in last_person_results:
                    fill = self._vio_tracker.person_state(pr, frame_w, frame_h)
                    _draw_person(
                        frame,
                        pr,
                        confirmed=pr.person_idx in confirmed_idxs,
                        fill=fill,
                    )

                n_pending = sum(
                    1 for pr in last_person_results if not pr.is_compliant and pr.person_idx not in confirmed_idxs
                )
                _draw_hud(
                    frame,
                    frame_idx,
                    live_fps,
                    last_yolo_ms,
                    n_persons=len(last_person_results),
                    n_pending=n_pending,
                    n_alerts=len(self._alerts),
                    vlm_calls=self._vlm_calls,
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

    # ── VLM validation ────────────────────────────────────────────────────────

    def _validate_and_alert(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        all_person_results: list[PersonPPEResult],
        confirmed: list[PersonPPEResult],
    ) -> FrameAlert:
        alert = FrameAlert(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            person_results=all_person_results,
        )

        if self.vlm is None:
            alert.vlm_confirmed = True  # trust YOLO + tracker alone
            return alert

        confirmed_by_vlm = False
        vlm_labels_all: list[str] = []

        for pr in confirmed:
            if not pr.needs_vlm_validation:
                confirmed_by_vlm = True
                continue

            crop = self._crop_person(frame, pr)
            tmp = "/tmp/_vlm_crop.jpg"
            cv.imwrite(tmp, crop)

            vlm_res = self.vlm.analyze(tmp)
            self._vlm_calls += 1

            if vlm_res.error:
                # VLM failed → conservatively confirm the YOLO alert
                confirmed_by_vlm = True
            else:
                viol_labels = [d.label for d in vlm_res.detections if d.is_violation]
                vlm_labels_all.extend(viol_labels)
                if viol_labels:
                    confirmed_by_vlm = True

        alert.vlm_confirmed = confirmed_by_vlm
        if vlm_labels_all:
            alert.vlm_response = str(vlm_labels_all)
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

        data = [
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
            for a in self._alerts
        ]
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_log_path.write_text(json.dumps(data, indent=2))
        print(f"Alert log → {self.alert_log_path}  ({len(data)} alerts)")


# ── Convenience factory ───────────────────────────────────────────────────────
def build_hybrid_tracker(
    yolo_model_path: str,
    vlm_backend: str = "none",  # "florence2" | "moondream" | "none"
    vlm_device: str = "cpu",
    persistence_frames: int = 5,
    window_frames: int = 8,
    skip_frames: int = 1,
    display: bool = True,
    save_output: str | None = None,
    alert_log: str | None = None,
    yolo_conf: float = 0.35,
) -> VideoSafetyTracker:
    """
    Quick-start factory.

    Example::

        tracker = build_hybrid_tracker(
            yolo_model_path="weights/best.onnx",
            vlm_backend="florence2",
            save_output="output/annotated.mp4",
            alert_log="output/alerts.json",
        )
        tracker.run("site_video.mp4")
    """
    from .analyzer import YOLOAnalyzer

    yolo = YOLOAnalyzer(model_path=yolo_model_path, conf_threshold=yolo_conf)

    vlm: BaseAnalyzer | None = None
    if vlm_backend == "florence2":
        from .analyzer import Florence2Analyzer

        vlm = Florence2Analyzer(device=vlm_device)
    elif vlm_backend == "moondream":
        from .analyzer import MoondreamAnalyzer

        vlm = MoondreamAnalyzer(device=vlm_device)
    elif vlm_backend != "none":
        msg = f"Unknown vlm_backend {vlm_backend!r}. Choose 'florence2', 'moondream', or 'none'."
        raise ValueError(msg)

    return VideoSafetyTracker(
        yolo_analyzer=yolo,
        vlm_validator=vlm,
        persistence_frames=persistence_frames,
        window_frames=window_frames,
        skip_frames=skip_frames,
        display=display,
        save_output=save_output,
        alert_log_path=alert_log,
    )
