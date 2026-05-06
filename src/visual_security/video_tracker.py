"""
Real-Time Video Safety Tracker — pipeline ibrida YOLO + PPEChecker + Moondream (opzionale).

Architettura:

  ┌────────────────┐  ogni N frame    ┌────────────────────────┐
  │  YOLO (veloce) │ ─ detections ──► │  PersonPPEChecker      │
  └────────────────┘                  │  (containment overlap) │
                                      └───────────┬────────────┘
                                                  │ PersonPPEResult list
                                      ┌───────────▼────────────┐
                                      │  VideoViolationTracker │
                                      │  (sliding window)      │
                                      └───────────┬────────────┘
                                                  │ violazioni confermate
                                      ┌───────────▼────────────┐
                                      │  Moondream2 (locale)   │  ← opzionale
                                      │  crop persona + prompt │
                                      └───────────┬────────────┘
                                                  │ FrameAlert
                                      ┌───────────▼────────────┐
                                      │  Output annotato       │
                                      │  (finestra / file)     │
                                      └────────────────────────┘
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2 as cv

from .person_ppe_checker import PersonPPEChecker, PersonPPEResult

if TYPE_CHECKING:
    import numpy as np

    from .analyzer import BaseAnalyzer, MoondreamAnalyzer, YOLOAnalyzer

# ── Alert ─────────────────────────────────────────────────────────────────────


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
        suffix = " [Moondream✓]" if self.vlm_confirmed else " [YOLO+tracker]"
        parts = " | ".join(p.summary() for p in viol)
        return f"[t={self.timestamp_s:.1f}s frame={self.frame_idx}] {len(viol)} violazione/i{suffix}: {parts}"


# ── Persistence tracker ───────────────────────────────────────────────────────


class VideoViolationTracker:
    """
    Sliding-window tracker per (cella_griglia × PPE_mancanti).

    Conferma una violazione quando appare in ≥ threshold dei
    ultimi `window` frame. Più robusto del "N consecutivi"
    perché tollera singole detection mancate.
    """

    def __init__(
        self,
        threshold_frames: int = 4,
        window: int = 7,
        grid: int = 6,
        cooldown_frames: int = 30,
    ):
        self.threshold = threshold_frames
        self.window = window
        self.grid = grid
        self.cooldown = cooldown_frames
        self._history: dict[str, deque] = {}
        self._cooldown: dict[str, int] = {}

    def _key(self, pr: PersonPPEResult, fw: int, fh: int) -> str:
        if pr.person_bbox:
            cx = min(int(((pr.person_bbox[0] + pr.person_bbox[2]) / 2) / max(fw, 1) * self.grid), self.grid - 1)
            cy = min(int(((pr.person_bbox[1] + pr.person_bbox[3]) / 2) / max(fh, 1) * self.grid), self.grid - 1)
        else:
            cx, cy = -1, -1
        return f"{cx}:{cy}:{'|'.join(sorted(pr.missing_ppe))}"

    def update(self, person_results: list[PersonPPEResult], fw: int, fh: int) -> list[PersonPPEResult]:
        active: set[str] = set()

        for pr in person_results:
            if pr.is_compliant:
                continue
            k = self._key(pr, fw, fh)
            active.add(k)
            if k not in self._history:
                self._history[k] = deque(maxlen=self.window)
            self._history[k].append(True)

        for k in list(self._history):
            if k not in active:
                self._history[k].append(False)
                if not any(self._history[k]):
                    self._history.pop(k, None)
                    self._cooldown.pop(k, None)

        for k in list(self._cooldown):
            self._cooldown[k] -= 1
            if self._cooldown[k] <= 0:
                del self._cooldown[k]

        confirmed: set[str] = set()
        for k, dq in self._history.items():
            if k not in self._cooldown and sum(dq) >= self.threshold:
                confirmed.add(k)
                self._cooldown[k] = self.cooldown

        return [pr for pr in person_results if not pr.is_compliant and self._key(pr, fw, fh) in confirmed]

    def fill(self, pr: PersonPPEResult, fw: int, fh: int) -> float:
        """Frazione della window riempita da positive (0–1)."""
        dq = self._history.get(self._key(pr, fw, fh))
        if not dq:
            return 0.0
        return sum(dq) / len(dq)


# ── Moondream VLM validator ───────────────────────────────────────────────────

_MOONDREAM_PROMPT_TEMPLATE = (
    "This image shows a construction worker cropped from a safety camera. "
    "The following PPE items were flagged as potentially MISSING: {missing}. "
    "For each item, answer whether it is truly MISSING (not visible at all) or PRESENT. "
    "Reply ONLY with a JSON object, keys are the PPE items, values are true=MISSING or false=PRESENT. "
    'Example: {{"Helmet": true, "Glove": false}}'
)


def _moondream_validate(
    vlm: MoondreamAnalyzer,
    image_source: str | np.ndarray,
    missing_ppe: list[str],
) -> tuple[bool, str]:
    """
    Usa MoondreamAnalyzer.query() con prompt specifico per i PPE mancanti.
    Ritorna (almeno_uno_confermato_mancante, risposta_raw).
    """
    prompt = _MOONDREAM_PROMPT_TEMPLATE.format(missing=", ".join(missing_ppe))
    try:
        answer = vlm.query(image_source, prompt)
        clean = answer.strip()
        if "```" in clean:
            clean = clean.split("```")[1].lstrip("json").strip()
        data = json.loads(clean)
        confirmed = any(data.get(k, False) for k in missing_ppe)
        return confirmed, answer
    except Exception as e:
        # In caso di errore parsing/modello: trust YOLO
        return True, f"[parse error: {e}]"


# ── Drawing ───────────────────────────────────────────────────────────────────

_C = {
    "green": (40, 200, 60),
    "red": (0, 45, 210),
    "yellow": (20, 190, 230),
    "white": (230, 230, 230),
    "dark": (20, 20, 20),
}


def _draw_person(img: np.ndarray, pr: PersonPPEResult, confirmed: bool, fill: float) -> None:
    if pr.person_bbox is None:
        return
    x1, y1, x2, y2 = (int(v) for v in pr.person_bbox)

    if pr.is_compliant:
        color, label, thick = _C["green"], "PPE OK", 2
    elif confirmed:
        color = _C["red"]
        label = f"ALERT: {', '.join(pr.missing_ppe)}"
        thick = 3
    else:
        # Colore che vira verso rosso man mano che fill cresce
        g = int(190 - fill * 150)
        color = (20, g, 230)
        label = f"[{int(fill * 100)}%] {', '.join(pr.missing_ppe)}"
        thick = 2

    cv.rectangle(img, (x1, y1), (x2, y2), color, thick)
    cv.putText(img, label, (x1, max(y1 - 8, 14)), cv.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv.LINE_AA)

    cy = y1 + 18
    for cat, req in PersonPPEChecker.FULL_PPE.items():
        found = pr.found_ppe.get(cat, 0)
        ok = found >= req
        c = _C["green"] if ok else _C["red"]
        mark = chr(0x2714) if ok else chr(0x2718)
        cv.putText(img, f"{mark} {cat} {found}/{req}", (x1 + 4, cy), cv.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv.LINE_AA)
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
    ov = img.copy()
    cv.rectangle(ov, (0, 0), (340, 135), _C["dark"], -1)
    cv.addWeighted(ov, 0.55, img, 0.45, 0, img)
    for i, line in enumerate(
        [
            f"Frame {frame_idx:>6}    FPS {fps:>5.1f}",
            f"YOLO  {yolo_ms:>7.1f} ms",
            f"Persone {n_persons:>3}   In attesa {n_pending:>3}",
            f"Alert   {n_alerts:>3}   VLM calls {vlm_calls:>4}",
        ]
    ):
        cv.putText(img, line, (8, 22 + i * 27), cv.FONT_HERSHEY_SIMPLEX, 0.52, _C["white"], 1, cv.LINE_AA)


# ── VideoSafetyTracker ────────────────────────────────────────────────────────


class VideoSafetyTracker:
    """
    Pipeline completa di tracking video per la sicurezza PPE.

    Parameters
    ----------
    yolo_analyzer       YOLOAnalyzer configurato.
    vlm_validator       MoondreamAnalyzer opzionale per validazione crop.
    ppe_checker         PersonPPEChecker (default: set PPE completo).
    persistence_frames  Frame positivi necessari nella window per confermare.
    window_frames       Larghezza sliding window (≥ persistence_frames).
    skip_frames         Esegui YOLO ogni N frame (1 = ogni frame).
    display             Mostra finestra OpenCV live.
    save_output         Percorso per il video annotato in output.
    alert_log_path      Percorso per il log JSON degli alert.
    max_alerts          Fermati dopo N alert confermati (0 = illimitato).
    verbose             Stampa debug su stdout ogni frame.
    """

    def __init__(
        self,
        yolo_analyzer: YOLOAnalyzer,
        vlm_validator: BaseAnalyzer | None = None,
        ppe_checker: PersonPPEChecker | None = None,
        persistence_frames: int = 4,
        window_frames: int = 7,
        skip_frames: int = 1,
        display: bool = True,
        save_output: str | Path | None = None,
        alert_log_path: str | Path | None = None,
        max_alerts: int = 0,
        verbose: bool = False,
    ):
        self.yolo = yolo_analyzer
        self.vlm = vlm_validator
        self.checker = ppe_checker or PersonPPEChecker()
        self._vio_tracker = VideoViolationTracker(
            threshold_frames=persistence_frames,
            window=max(window_frames, persistence_frames),
        )
        self.skip_frames = max(1, skip_frames)
        self.display = display
        self.save_output = Path(save_output) if save_output else None
        self.alert_log_path = Path(alert_log_path) if alert_log_path else None
        self.max_alerts = max_alerts
        self.verbose = verbose

        self._vlm_calls = 0
        self._alerts: list[FrameAlert] = []

    # ── Entry point ────────────────────────────────────────────────────────────
    def run(self, source: str | int = 0) -> list[FrameAlert]:
        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            msg = f"Impossibile aprire la sorgente video: {source!r}"
            raise RuntimeError(msg)

        fps_src = cap.get(cv.CAP_PROP_FPS) or 25.0
        fw = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.save_output:
            self.save_output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv.VideoWriter(
                str(self.save_output),
                cv.VideoWriter_fourcc(*"mp4v"),
                fps_src,
                (fw, fh),
            )

        frame_idx = 0
        last_ppr: list[PersonPPEResult] = []
        last_ms = 0.0
        fps_buf = deque(maxlen=30)
        t_last = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                t_now = time.perf_counter()
                fps_buf.append(1.0 / max(t_now - t_last, 1e-6))
                t_last = t_now
                live_fps = sum(fps_buf) / len(fps_buf)

                # ── YOLO + PPE check ─────────────────────────────────────────
                if (frame_idx - 1) % self.skip_frames == 0:
                    yolo_res = self.yolo.analyze(frame)  # ndarray diretto, niente disco
                    last_ms = yolo_res.inference_time_ms

                    if yolo_res.error:
                        if self.verbose:
                            print(f"[frame {frame_idx}] YOLO error: {yolo_res.error}")
                    else:
                        # Passa le dimensioni reali del frame per denormalizzare bbox
                        last_ppr = self.checker.check(yolo_res.detections, fw, fh)

                        if self.verbose:
                            print(f"\n[frame {frame_idx}] {len(yolo_res.detections)} det → {len(last_ppr)} persone")
                            for pr in last_ppr:
                                print(f"  {pr.summary()}")

                # ── Persistence filtering ────────────────────────────────────
                confirmed = self._vio_tracker.update(last_ppr, fw, fh)

                # ── VLM validation + alert ───────────────────────────────────
                if confirmed:
                    alert = self._make_alert(frame, frame_idx, cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0, last_ppr, confirmed)
                    self._alerts.append(alert)
                    print(alert.summary())
                    if self.max_alerts and len(self._alerts) >= self.max_alerts:
                        break

                # ── Annotazione ──────────────────────────────────────────────
                confirmed_idxs = {pr.person_idx for pr in confirmed}
                for pr in last_ppr:
                    fill = self._vio_tracker.fill(pr, fw, fh)
                    _draw_person(frame, pr, pr.person_idx in confirmed_idxs, fill)

                n_pending = sum(1 for pr in last_ppr if not pr.is_compliant and pr.person_idx not in confirmed_idxs)
                _draw_hud(frame, frame_idx, live_fps, last_ms, len(last_ppr), n_pending, len(self._alerts), self._vlm_calls)

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
                self._save_log()

        return self._alerts

    # ── VLM ───────────────────────────────────────────────────────────────────

    def _make_alert(self, frame, frame_idx, ts, all_ppr, confirmed) -> FrameAlert:
        alert = FrameAlert(frame_idx=frame_idx, timestamp_s=ts, person_results=all_ppr)
        if self.vlm is None:
            alert.vlm_confirmed = True
            return alert

        any_confirmed = False
        vlm_resp_parts: list[str] = []
        for pr in confirmed:
            crop = self._crop(frame, pr)
            if crop is None:
                any_confirmed = True
                continue
            # crop è un np.ndarray BGR — niente disco, passato direttamente
            ok, resp = _moondream_validate(self.vlm, crop, pr.missing_ppe)
            self._vlm_calls += 1
            if ok:
                any_confirmed = True
            vlm_resp_parts.append(f"Person#{pr.person_idx}: {resp}")

        alert.vlm_confirmed = any_confirmed
        alert.vlm_response = " | ".join(vlm_resp_parts) or None
        return alert

    @staticmethod
    def _crop(frame: np.ndarray, pr: PersonPPEResult):
        if pr.person_bbox is None:
            return None
        h, w = frame.shape[:2]
        x1 = max(0, int(pr.person_bbox[0]))
        y1 = max(0, int(pr.person_bbox[1]))
        x2 = min(w, int(pr.person_bbox[2]))
        y2 = min(h, int(pr.person_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    # ── Log ───────────────────────────────────────────────────────────────────

    def _save_log(self):
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
        print(f"Log alert → {self.alert_log_path} ({len(data)} alert)")


# ── Factory ───────────────────────────────────────────────────────────────────


def build_hybrid_tracker(
    yolo_model_path: str,
    vlm_backend: str = "none",  # "moondream" | "none"
    vlm_device: str = "cpu",
    persistence_frames: int = 4,
    window_frames: int = 7,
    skip_frames: int = 1,
    display: bool = True,
    save_output: str | None = None,
    alert_log: str | None = None,
    yolo_conf: float = 0.30,
    verbose: bool = False,
) -> VideoSafetyTracker:
    """
    Factory rapida.

    Esempio::

        tracker = build_hybrid_tracker(
            yolo_model_path="weights/best.onnx",
            save_output="output/annotated.mp4",
            alert_log="output/alerts.json",
            verbose=True,  # stampa detection ogni frame
        )
        tracker.run("test.mp4")
    """
    from .analyzer import YOLOAnalyzer

    yolo = YOLOAnalyzer(model_path=yolo_model_path, conf_threshold=yolo_conf)

    vlm: BaseAnalyzer | None = None
    if vlm_backend == "moondream":
        from .analyzer import MoondreamAnalyzer

        vlm = MoondreamAnalyzer(device=vlm_device)
    elif vlm_backend not in ("none", ""):
        msg = f"vlm_backend={vlm_backend!r} non riconosciuto. Usa 'moondream' o 'none'."
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
        verbose=verbose,
    )
