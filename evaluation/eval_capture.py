"""
Cattura TUTTE le detection grezze dei due detector sul subset SH17, a soglia
bassa e con il filtro per-classe DISATTIVATO. Le metriche (mAP, F1, sweep soglia)
si calcolano poi OFFLINE su questo dump (eval_metrics.py) senza rieseguire i modelli.
"""
import json
import sys
import time
from pathlib import Path

import cv2 as cv

sys.path.insert(0, "src")
from visual_security.analyzer import GroundingDinoAnalyzer, OmDetTurboAnalyzer  # noqa: E402

SUBSET = Path("evaluation/sh17_subset")
KEEP = {"Person", "Helmet", "Vest", "Glasses", "Glove", "Shoe", "Cigarette"}

gt = json.load(open(SUBSET / "ground_truth.json", encoding="utf-8"))
print(f"Immagini: {len(gt)}")

# Soglia bassissima + niente filtro per-classe: catturiamo tutto lo spettro di
# confidence, cosi' offline possiamo tracciare la curva P/R completa (per l'AP).
detectors = {
    "grounding-dino": GroundingDinoAnalyzer(conf_threshold=0.05, text_threshold=0.10),
    "omdet-turbo": OmDetTurboAnalyzer(conf_threshold=0.05),
}

for name, det in detectors.items():
    det.class_conf = {}  # nessun floor per-classe: filtriamo offline
    print(f"\n[{name}] cattura...", flush=True)
    det.analyze(cv.imread(str(SUBSET / "images" / gt[0]["file"])))  # warmup
    out = {}
    t0 = time.perf_counter()
    for i, item in enumerate(gt):
        img = cv.imread(str(SUBSET / "images" / item["file"]))
        res = det.analyze(img)
        out[item["file"]] = [
            [d.label, round(d.bbox[0], 1), round(d.bbox[1], 1), round(d.bbox[2], 1), round(d.bbox[3], 1),
             round(float(d.confidence), 4)]
            for d in res.detections
            if d.label in KEEP
        ]
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(gt)}", flush=True)
    json.dump(out, open(SUBSET.parent / f"raw_detections_{name}.json", "w"), indent=0)
    n_det = sum(len(v) for v in out.values())
    print(f"[{name}] {n_det} detection salvate in {time.perf_counter() - t0:.0f}s", flush=True)

print("\nCattura completata.")
