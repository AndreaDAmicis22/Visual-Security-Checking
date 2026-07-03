"""
Real-Time Video Safety Tracker — pipeline YOLO + PPEChecker + VLM locale.

Architecture:

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
                                      │  VLM locale (in-proc)  │  ← opzionale
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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2 as cv

from .person_ppe_checker import PersonPPEChecker, PersonPPEResult

if TYPE_CHECKING:
    import numpy as np

    from .analyzer import YOLOAnalyzer
    from .vlm_validator import LocalVLMValidator


# ── Alert ─────────────────────────────────────────────────────────────────────
@dataclass
class FrameAlert:
    frame_idx: int
    timestamp_s: float
    person_results: list[PersonPPEResult]
    # None = validazione VLM ancora in corso (thread in background);
    # True/False = esito del VLM; False anche quando il VLM non è configurato.
    vlm_confirmed: bool | None = False
    vlm_response: str | None = None
    # Verdetto VLM per singola persona (person_idx -> confermato/scagionato),
    # a differenza di vlm_confirmed che e' aggregato sull'intero alert.
    vlm_person_verdicts: dict[int, bool] = field(default_factory=dict)

    @property
    def violations(self) -> list[PersonPPEResult]:
        return [p for p in self.person_results if not p.is_compliant]

    def summary(self) -> str:
        viol = self.violations
        if self.vlm_confirmed is None:
            source = "[YOLO+Tracker | VLM pending]"
        elif self.vlm_confirmed:
            source = "[YOLO+Tracker+VLM OK]"
        else:
            source = "[YOLO+Tracker]"
        parts = " | ".join(p.summary() for p in viol)
        return f"[t={self.timestamp_s:.1f}s frame={self.frame_idx}] {len(viol)} violazione/i {source}: {parts}"


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

    def key(self, pr: PersonPPEResult, fw: int, fh: int) -> str:
        """Chiave pubblica (cella_griglia × PPE_mancanti) — usata anche per cache-are i verdetti VLM."""
        return self._key(pr, fw, fh)

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


# ── Drawing ───────────────────────────────────────────────────────────────────
_C = {
    "green": (40, 200, 60),
    "red": (0, 45, 210),
    "yellow": (20, 190, 230),
    "white": (230, 230, 230),
    "dark": (20, 20, 20),
    "purple": (180, 60, 180),
}


def _draw_person(
    img: np.ndarray, pr: PersonPPEResult, confirmed: bool, fill: float, vlm_verdict: bool | None = None
) -> None:
    if pr.person_bbox is None:
        return
    x1, y1, x2, y2 = (int(v) for v in pr.person_bbox)

    if pr.is_compliant:
        color, label, thick = _C["green"], "PPE OK", 2
    elif confirmed and vlm_verdict is False:
        # Il tracker persiste a segnalarla, ma il VLM ha gia' scagionato la persona.
        color = _C["purple"]
        label = f"VLM: PPE presente? {', '.join(pr.missing_ppe)}"
        thick = 2
    elif confirmed:
        color = _C["red"]
        label = f"ALERT: {', '.join(pr.missing_ppe)}" + (" [VLM OK]" if vlm_verdict else "")
        thick = 3
    else:
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
    vlm_validator       LocalVLMValidator opzionale per validazione crop.
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
        vlm_validator: LocalVLMValidator | None = None,
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
        # Il VLM (lento su CPU) gira su un singolo worker fuori dal loop dei
        # frame, così il video/display non si blocca a ogni alert.
        self._vlm_executor: ThreadPoolExecutor | None = None
        self._vlm_futures: list[Future] = []
        # Cache dei verdetti VLM per chiave di violazione (VideoViolationTracker.key),
        # cosi' i frame SUCCESSIVI alla risposta async possono disegnare il box
        # gia' corretto (il frame in cui l'alert e' stato emesso e' gia' scritto
        # su disco/mostrato prima che il VLM risponda, e non e' recuperabile).
        self._vlm_verdicts: dict[str, bool] = {}

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

        if self.vlm is not None:
            self._vlm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")

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
                    yolo_res = self.yolo.analyze(frame)
                    last_ms = yolo_res.inference_time_ms

                    if yolo_res.error:
                        if self.verbose:
                            print(f"[frame {frame_idx}] YOLO error: {yolo_res.error}")
                    else:
                        last_ppr = self.checker.check(yolo_res.detections, fw, fh)

                        if self.verbose:
                            print(f"\n[frame {frame_idx}] {len(yolo_res.detections)} det -> {len(last_ppr)} persone")
                            for pr in last_ppr:
                                print(f"  {pr.summary()}")

                # ── Persistence filtering ────────────────────────────────────
                confirmed = self._vio_tracker.update(last_ppr, fw, fh)

                # ── Alert + VLM validation (async) ───────────────────────────
                if confirmed:
                    ts = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
                    alert = self._emit_alert(frame, frame_idx, ts, last_ppr, confirmed)
                    self._alerts.append(alert)
                    if self.max_alerts and len(self._alerts) >= self.max_alerts:
                        break

                # ── Annotazione ──────────────────────────────────────────────
                confirmed_idxs = {pr.person_idx for pr in confirmed}
                for pr in last_ppr:
                    fill = self._vio_tracker.fill(pr, fw, fh)
                    is_confirmed = pr.person_idx in confirmed_idxs
                    vlm_verdict = self._vlm_verdicts.get(self._vio_tracker.key(pr, fw, fh)) if is_confirmed else None
                    _draw_person(frame, pr, is_confirmed, fill, vlm_verdict)

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
            # Attendi le validazioni VLM ancora in corso prima di loggare.
            if self._vlm_executor is not None:
                if self._vlm_futures:
                    print(f"Attendo {sum(not f.done() for f in self._vlm_futures)} validazione/i VLM in corso ...")
                self._vlm_executor.shutdown(wait=True)
                self._vlm_executor = None
            if self.alert_log_path:
                self._save_log()

        return self._alerts

    # ── VLM ───────────────────────────────────────────────────────────────────
    def _emit_alert(self, frame, frame_idx, ts, all_ppr, confirmed) -> FrameAlert:
        """
        Crea subito il FrameAlert. Se il VLM è configurato la validazione parte
        su un thread in background (stato vlm_confirmed=None finché non finisce),
        così il loop dei frame non si blocca.
        """
        alert = FrameAlert(frame_idx=frame_idx, timestamp_s=ts, person_results=all_ppr)

        if self.vlm is None or self._vlm_executor is None:
            alert.vlm_confirmed = False
            alert.vlm_response = "VLM non configurato -> decisione da YOLO+tracker"
            print(alert.summary())
            return alert

        # Copia i crop ORA: `frame` viene sovrascritto al prossimo giro.
        # Portiamo anche la chiave del tracker, cosi' _run_vlm puo' aggiornare
        # la cache dei verdetti usata dai frame successivi per il disegno.
        fh, fw = frame.shape[:2]
        crops: list[tuple[PersonPPEResult, np.ndarray, str]] = []
        for pr in confirmed:
            crop = self._crop(frame, pr, pad_ratio=0.30)
            if crop is not None:
                crops.append((pr, crop.copy(), self._vio_tracker.key(pr, fw, fh)))

        alert.vlm_confirmed = None  # pending
        print(alert.summary())
        future = self._vlm_executor.submit(self._run_vlm, alert, crops)
        self._vlm_futures.append(future)
        return alert

    def _run_vlm(self, alert: FrameAlert, crops: list[tuple[PersonPPEResult, np.ndarray, str]]) -> None:
        """Eseguito nel worker VLM: valida i crop e aggiorna l'alert in-place."""
        confirmed_count = 0
        parts: list[str] = []
        for pr, crop, key in crops:
            ok, resp = self.vlm.validate_missing_ppe(crop, pr.missing_ppe)
            self._vlm_calls += 1
            alert.vlm_person_verdicts[pr.person_idx] = ok
            self._vlm_verdicts[key] = ok
            if ok:
                confirmed_count += 1
            parts.append(f"Person#{pr.person_idx}: {resp}")

        alert.vlm_confirmed = confirmed_count > 0
        alert.vlm_response = " | ".join(parts) or None
        verdict = "CONFERMATA" if alert.vlm_confirmed else "scartata (falso positivo?)"
        print(f"  -> VLM frame {alert.frame_idx}: {verdict} - {alert.vlm_response}")

    @staticmethod
    def _crop(frame: np.ndarray, pr: PersonPPEResult, pad_ratio: float = 0.30) -> np.ndarray | None:
        """Crop the person from the frame with padding for VLM context."""
        if pr.person_bbox is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = pr.person_bbox
        w_orig = x2 - x1
        h_orig = y2 - y1
        pad_w = w_orig * pad_ratio
        pad_h = h_orig * pad_ratio
        nx1 = max(0, int(x1 - pad_w))
        ny1 = max(0, int(y1 - pad_h))
        nx2 = min(w, int(x2 + pad_w))
        ny2 = min(h, int(y2 + pad_h))
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return frame[ny1:ny2, nx1:nx2]

    # ── Log ───────────────────────────────────────────────────────────────────
    def _save_log(self):
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
                        "vlm_confirmed": a.vlm_person_verdicts.get(p.person_idx),
                    }
                    for p in a.violations
                ],
            }
            for a in self._alerts
        ]
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_log_path.write_text(json.dumps(data, indent=2))
        print(f"Log alert -> {self.alert_log_path} ({len(data)} alert)")


# ── Factory ───────────────────────────────────────────────────────────────────
def build_tracker(
    yolo_model_path: str,
    vlm_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
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
    Factory function per creare il tracker.

    Parameters
    ----------
    vlm_model : str
        ID HuggingFace del VLM locale per la validazione (in-process, no server).
        "none" per disabilitare. Default: "HuggingFaceTB/SmolVLM-500M-Instruct".
        Alternativa più accurata: "HuggingFaceTB/SmolVLM2-2.2B-Instruct".

    Example
    -------
        tracker = build_tracker(
            yolo_model_path="weights/best.onnx",
            vlm_model="HuggingFaceTB/SmolVLM-500M-Instruct",
            save_output="output/annotated.mp4",
            alert_log="output/alerts.json",
            verbose=True,
        )
        tracker.run("test.mp4")
    """
    from .analyzer import YOLOAnalyzer

    yolo = YOLOAnalyzer(model_path=yolo_model_path, conf_threshold=yolo_conf)

    vlm: LocalVLMValidator | None = None
    if vlm_model not in ("none", ""):
        from .vlm_validator import LocalVLMValidator

        if not LocalVLMValidator.is_available():
            print("[WARN] torch/transformers non installati -> VLM disabilitato.")
            print("       Installa con: pip install torch transformers pillow")
        else:
            print(f"[VLM] caricamento backend locale '{vlm_model}' ...")
            t0 = time.perf_counter()
            candidate = LocalVLMValidator(model_id=vlm_model)
            if candidate.load():
                vlm = candidate
                print(f"[VLM] pronto in {time.perf_counter() - t0:.1f}s.")
            else:
                print(f"[WARN] caricamento VLM fallito ({candidate.load_error}) -> VLM disabilitato.")

    return VideoSafetyTracker(
        yolo_analyzer=yolo,
        vlm_validator=vlm,
        persistence_frames=persistence_frames,
        window_frames=max(window_frames, persistence_frames),
        skip_frames=skip_frames,
        display=display,
        save_output=save_output,
        alert_log_path=alert_log,
        verbose=verbose,
    )
