"""
Statistiche TEMPORALI della pipeline sui video, per entrambi i detector.

Rilancia il tracker su ppe_video.mp4 con la strumentazione per-frame attivata
(`collect_stats=True`) e con gli STESSI parametri delle demo (skip=8,
persistence=3, window=6, ppe_memory=50). Non salva video (gia' prodotti):
serve solo a misurare il comportamento nel tempo.

Metriche:
- Latenza detection (solo sui frame in cui il detector gira): media/mediana/p95/max.
- Throughput effettivo end-to-end (frame totali / wall time) = FPS reali della pipeline.
- Stabilita' identita': track creati (meno track per lo stesso numero di persone
  reali = identita' piu' stabile, meno "ID switch").
- Carico medio: persone per frame.
- Profilo alert: n. alert confermati, tempo al primo alert, span temporale,
  frequenza delle violazioni per etichetta.
"""
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from visual_security.video_tracker import build_tracker  # noqa: E402

VIDEO = "ppe_video.mp4"
DETECTORS = ["grounding-dino", "omdet-turbo"]
OUT = Path("evaluation/temporal_stats_results.json")

results = {}
for name in DETECTORS:
    print(f"\n{'=' * 60}\n[{name}] statistiche temporali\n{'=' * 60}", flush=True)
    tracker = build_tracker(
        detector=name,
        persistence_frames=3,
        window_frames=6,
        skip_frames=8,
        ppe_memory_frames=50,
        display=False,
        save_output=None,
        alert_log=None,
        verbose=False,
    )
    tracker.collect_stats = True
    t0 = time.perf_counter()
    alerts = tracker.run(VIDEO)
    wall = time.perf_counter() - t0

    fl = tracker.frame_log
    n_frames = len(fl)
    det_ms = [r["det_ms"] for r in fl if r["det_ran"] and r["det_ms"] > 0]
    persons = [r["n_persons"] for r in fl]

    # profilo alert
    alert_ts = [a.timestamp_s for a in alerts]
    viol_labels = Counter()
    for a in alerts:
        for pr in a.violations:
            viol_labels.update(pr.violation_labels)

    def pct(v, p):
        return round(sorted(v)[min(len(v) - 1, int(len(v) * p))], 1) if v else 0.0

    results[name] = {
        "model_id": tracker.detector.model_id,
        "n_frames": n_frames,
        "wall_seconds": round(wall, 1),
        "effective_fps": round(n_frames / wall, 3) if wall else 0.0,
        "detector_runs": len(det_ms),
        "detection_latency_ms": {
            "mean": round(st.mean(det_ms), 1) if det_ms else 0.0,
            "median": round(st.median(det_ms), 1) if det_ms else 0.0,
            "p95": pct(det_ms, 0.95),
            "min": round(min(det_ms), 1) if det_ms else 0.0,
            "max": round(max(det_ms), 1) if det_ms else 0.0,
        },
        "persons_per_frame_mean": round(st.mean(persons), 2) if persons else 0.0,
        "tracks_created": tracker.tracks_created,
        "alerts": {
            "n_confirmed": len(alerts),
            "time_to_first_s": round(min(alert_ts), 2) if alert_ts else None,
            "last_alert_s": round(max(alert_ts), 2) if alert_ts else None,
            "span_s": round(max(alert_ts) - min(alert_ts), 2) if alert_ts else 0.0,
            "violation_label_freq": dict(viol_labels),
        },
    }
    r = results[name]
    print(f"[{name}] FPS eff={r['effective_fps']} | det lat mean={r['detection_latency_ms']['mean']}ms "
          f"p95={r['detection_latency_ms']['p95']}ms | track={r['tracks_created']} | "
          f"alert={r['alerts']['n_confirmed']} (1o a {r['alerts']['time_to_first_s']}s)", flush=True)

json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2)
print(f"\nStatistiche temporali salvate in {OUT}")
