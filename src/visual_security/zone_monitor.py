"""
zone_monitor.py - Aree vietate: rileva persone dentro zone poligonali proibite.

Non serve alcun modello: la posizione della persona arriva dal detector
(bbox), l'appartenenza alla zona e' pura geometria. Il punto testato e' il
**punto-piedi** (centro del lato inferiore della bbox): e' la proiezione a
terra della persona, molto piu' affidabile del centro-bbox quando la zona
e' disegnata sul pavimento e la persona e' alta nell'inquadratura.

Le zone si definiscono in un file JSON:

    {
      "zones": [
        {
          "name": "Area gru",
          "polygon": [[0.10, 0.55], [0.45, 0.55], [0.45, 0.95], [0.10, 0.95]],
          "normalized": true
        },
        {
          "name": "Deposito",
          "polygon": [[900, 100], [1250, 100], [1250, 400], [900, 400]]
        }
      ]
    }

- `polygon`: lista di vertici [x, y], almeno 3.
- `normalized`: true se le coordinate sono frazioni 0-1 del frame (consigliato:
  indipendenti dalla risoluzione). Default false = pixel assoluti.

Come per i PPE, una violazione zona viene confermata solo se persiste
(sliding window + cooldown) — una persona che sfiora il bordo per un solo
frame non genera un alert.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np

if TYPE_CHECKING:
    from .person_ppe_checker import PersonPPEResult


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class RestrictedZone:
    name: str
    polygon: list[tuple[float, float]]  # vertici come definiti nel file
    normalized: bool = False

    def pixel_polygon(self, fw: int, fh: int) -> np.ndarray:
        """Vertici in pixel (int32, shape Nx2) per il frame corrente."""
        pts = [(x * fw, y * fh) if self.normalized else (x, y) for x, y in self.polygon]
        return np.array(pts, dtype=np.int32)


@dataclass
class ZoneViolation:
    zone_name: str
    person_idx: int
    track_id: int | None
    foot_point: tuple[int, int]
    person_bbox: tuple | None = None
    confirmed: bool = False  # True quando supera la persistenza
    fill: float = 0.0  # frazione della sliding window (0-1)

    def summary(self) -> str:
        tid = f"T{self.track_id}" if self.track_id is not None else f"#{self.person_idx}"
        state = "CONFERMATA" if self.confirmed else f"in accumulo {int(self.fill * 100)}%"
        return f"Persona {tid} in zona vietata '{self.zone_name}' ({state})"


# ── Monitor ───────────────────────────────────────────────────────────────────
class ZoneMonitor:
    """
    Verifica frame per frame se il punto-piedi delle persone cade in una zona
    vietata, con la stessa logica di persistenza degli alert PPE.

    Parameters
    ----------
    zones : list[RestrictedZone]
    persistence_frames : int
        Frame positivi nella window per confermare l'ingresso in zona.
    window_frames : int
        Ampiezza della sliding window.
    cooldown_frames : int
        Dopo una conferma, la stessa (persona, zona) non ri-scatta per N frame.
    """

    def __init__(
        self,
        zones: list[RestrictedZone],
        persistence_frames: int = 4,
        window_frames: int = 7,
        cooldown_frames: int = 60,
    ):
        self.zones = zones
        self.persistence = persistence_frames
        self.window = max(window_frames, persistence_frames)
        self.cooldown = cooldown_frames
        self._history: dict[str, deque] = {}
        self._cooldown: dict[str, int] = {}

    # ── Loading ───────────────────────────────────────────────────────────────
    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> ZoneMonitor:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        zones = [
            RestrictedZone(
                name=z["name"],
                polygon=[tuple(p) for p in z["polygon"]],
                normalized=bool(z.get("normalized", False)),
            )
            for z in data.get("zones", [])
        ]
        if not zones:
            msg = f"Nessuna zona definita in {path}"
            raise ValueError(msg)
        for z in zones:
            if len(z.polygon) < 3:
                msg = f"La zona '{z.name}' ha meno di 3 vertici"
                raise ValueError(msg)
        return cls(zones, **kwargs)

    # ── Core ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _foot_point(bbox: tuple) -> tuple[int, int]:
        x1, _, x2, y2 = bbox
        return (int((x1 + x2) / 2), int(y2))

    def _key(self, zone: RestrictedZone, pr: PersonPPEResult) -> str:
        pid = f"T{pr.track_id}" if pr.track_id is not None else f"P{pr.person_idx}"
        return f"{pid}@{zone.name}"

    def check(self, person_results: list[PersonPPEResult], fw: int, fh: int) -> list[ZoneViolation]:
        """
        Ritorna le violazioni zona del frame corrente (confermate e non).
        Da chiamare a ogni frame in cui il detector ha girato.
        """
        violations: list[ZoneViolation] = []
        active: set[str] = set()

        for zone in self.zones:
            poly = zone.pixel_polygon(fw, fh).astype(np.float32)
            for pr in person_results:
                if pr.person_bbox is None:
                    continue
                foot = self._foot_point(pr.person_bbox)
                inside = cv.pointPolygonTest(poly, (float(foot[0]), float(foot[1])), False) >= 0
                if not inside:
                    continue

                k = self._key(zone, pr)
                active.add(k)
                if k not in self._history:
                    self._history[k] = deque(maxlen=self.window)
                self._history[k].append(True)

                violations.append(
                    ZoneViolation(
                        zone_name=zone.name,
                        person_idx=pr.person_idx,
                        track_id=pr.track_id,
                        foot_point=foot,
                        person_bbox=pr.person_bbox,
                    )
                )

        # Aggiorna history delle chiavi non attive e i cooldown.
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

        # Marca confermate quelle sopra soglia (e avvia il cooldown).
        for v in violations:
            k = f"{'T' + str(v.track_id) if v.track_id is not None else 'P' + str(v.person_idx)}@{v.zone_name}"
            dq = self._history.get(k)
            if not dq:
                continue
            v.fill = sum(dq) / len(dq)
            if k not in self._cooldown and sum(dq) >= self.persistence:
                v.confirmed = True
                self._cooldown[k] = self.cooldown

        return violations

    # ── Drawing ───────────────────────────────────────────────────────────────
    def draw(self, img: np.ndarray, active_zones: set[str] | None = None) -> None:
        """
        Disegna le zone sul frame: overlay semitrasparente + bordo.
        Le zone in `active_zones` (violazione in corso) sono evidenziate.
        """
        fh, fw = img.shape[:2]
        overlay = img.copy()
        for zone in self.zones:
            poly = zone.pixel_polygon(fw, fh)
            hot = active_zones is not None and zone.name in active_zones
            color = (0, 45, 210) if hot else (30, 140, 240)  # BGR: rosso se attiva
            cv.fillPoly(overlay, [poly], color)
            cv.polylines(img, [poly], isClosed=True, color=color, thickness=2, lineType=cv.LINE_AA)
            # Etichetta sul primo vertice
            x0, y0 = poly[0]
            cv.putText(
                img,
                f"ZONA VIETATA: {zone.name}",
                (int(x0) + 4, max(int(y0) - 6, 14)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv.LINE_AA,
            )
        cv.addWeighted(overlay, 0.18, img, 0.82, 0, img)
