"""
person_tracker.py - Identita' persistente delle persone + memoria PPE temporale.

Due problemi risolti insieme:

1. Identita': PersonPPEChecker lavora frame per frame, quindi "Persona#2" di un
   frame non ha alcun legame con "Persona#2" del frame successivo. Questo
   tracker associa le bbox persona tra frame consecutivi (matching greedy per
   IoU) e assegna un `track_id` stabile, che VideoViolationTracker usa come
   chiave di violazione (al posto della cella di griglia: una persona che
   cammina non resetta piu' il conteggio di persistenza) e che rende la cache
   dei verdetti VLM davvero per-persona.

2. Memoria PPE: YOLO perde facilmente gli oggetti piccoli (Glove/Shoe) per
   occlusione o motion blur, producendo violazioni intermittenti "fantasma".
   Per ogni track viene ricordato il conteggio PPE osservato di recente: un
   PPE visto negli ultimi `ppe_memory_frames` frame e' considerato ancora
   presente anche se il frame corrente non lo rileva. `found_ppe` e
   `missing_ppe` dei PersonPPEResult vengono riscritti di conseguenza
   (il guanto visto un secondo fa non e' sparito davvero).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .person_ppe_checker import REQUIRED_PPE_COUNTS, PersonPPEResult, _iou


@dataclass
class _Track:
    track_id: int
    bbox: tuple
    last_seen: int  # frame_idx dell'ultimo match
    # cat -> (conteggio osservato, frame dell'osservazione)
    ppe_evidence: dict[str, tuple[int, int]] = field(default_factory=dict)


class PersonTracker:
    """
    Tracker IoU-based con memoria PPE per persona.

    Parameters
    ----------
    iou_threshold : float
        IoU minimo per associare una persona a un track esistente.
    max_age_frames : int
        Frame senza match dopo i quali un track viene eliminato.
    ppe_memory_frames : int
        Finestra di validita' dell'evidenza PPE: un PPE osservato entro
        questa distanza (in frame) e' considerato ancora presente.
        Default 48 (~2s a 24fps).
    required_ppe : dict[str, int] | None
        Quantita' richiesta per categoria (default: set completo, come
        PersonPPEChecker).
    vlm_trigger_missing : set[str] | None
        Stessa semantica di PersonPPEChecker: quali categorie mancanti
        fanno scattare la validazione VLM (None = qualunque).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age_frames: int = 30,
        ppe_memory_frames: int = 48,
        required_ppe: dict[str, int] | None = None,
        vlm_trigger_missing: set[str] | None = None,
    ):
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self.ppe_memory_frames = ppe_memory_frames
        self.required_ppe = required_ppe or dict(REQUIRED_PPE_COUNTS)
        self.vlm_trigger_missing = vlm_trigger_missing
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def update(self, person_results: list[PersonPPEResult], frame_idx: int) -> list[PersonPPEResult]:
        """
        Associa le persone del frame ai track, aggiorna l'evidenza PPE e
        riscrive in-place found_ppe/missing_ppe/needs_vlm_validation.
        Va chiamato solo sui frame in cui YOLO ha realmente girato.
        """
        with_bbox = [pr for pr in person_results if pr.person_bbox is not None]

        # ── Elimina i track scaduti PRIMA del matching: un track oltre
        #    max_age non deve poter catturare una persona nuova ─────────────────
        expired = [tid for tid, tr in self._tracks.items() if frame_idx - tr.last_seen > self.max_age_frames]
        for tid in expired:
            del self._tracks[tid]

        # ── Matching greedy: coppie (track, persona) per IoU decrescente ──────
        pairs: list[tuple[float, int, int]] = []
        for tid, tr in self._tracks.items():
            for j, pr in enumerate(with_bbox):
                iou = _iou(tr.bbox, tuple(pr.person_bbox))
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, j))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_persons: set[int] = set()
        for _, tid, j in pairs:
            if tid in matched_tracks or j in matched_persons:
                continue
            matched_tracks.add(tid)
            matched_persons.add(j)
            self._assign(self._tracks[tid], with_bbox[j], frame_idx)

        # ── Nuovi track per le persone non associate ──────────────────────────
        for j, pr in enumerate(with_bbox):
            if j in matched_persons:
                continue
            track = _Track(track_id=self._next_id, bbox=tuple(pr.person_bbox), last_seen=frame_idx)
            self._next_id += 1
            self._tracks[track.track_id] = track
            self._assign(track, pr, frame_idx)

        return person_results

    def _assign(self, track: _Track, pr: PersonPPEResult, frame_idx: int) -> None:
        track.bbox = tuple(pr.person_bbox)
        track.last_seen = frame_idx
        pr.track_id = track.track_id

        # Il conteggio ricordato sopravvive finche' la memoria e' valida e
        # viene rinfrescato quando l'osservazione corrente e' pari o migliore;
        # quando la memoria scade si degrada al conteggio corrente.
        for cat in self.required_ppe:
            cur = pr.found_ppe.get(cat, 0)
            prev = track.ppe_evidence.get(cat)
            if prev is None or cur >= prev[0] or frame_idx - prev[1] > self.ppe_memory_frames:
                track.ppe_evidence[cat] = (cur, frame_idx)

        effective = {cat: track.ppe_evidence[cat][0] for cat in self.required_ppe}
        pr.found_ppe = effective
        pr.missing_ppe = [cat for cat, req in self.required_ppe.items() if effective[cat] < req]
        pr.needs_vlm_validation = (
            bool(pr.missing_ppe)
            if self.vlm_trigger_missing is None
            else bool(set(pr.missing_ppe) & self.vlm_trigger_missing)
        )
