"""
Valutazione statica dei due detector open-vocabulary su un subset bilanciato
di SH17 (vedi build_subset.py).

Metodo:
- Soglie IDENTICHE al pipeline video: `build_detector(name)` usa la soglia base
  del backend (GD 0.35 / OmDet 0.30) + le soglie per-classe `DETECTION_CONF`
  (Glasses 0.45, Cigarette 0.50). Cosi' il test statico riflette la config reale.
- Matching predizione<->GT per classe con IoU >= IOU_THR (greedy, confidence-desc).
  TP = match; FP = predizione non matchata; FN = GT non matchata.
- Metriche per classe: Precision, Recall, F1 (al punto operativo, non AP: interessa
  la resa del sistema COSI' COM'E' configurato).
- I check geometrici di PersonPPEChecker (occhiali in fascia-volto, dimensione
  sigaretta) NON sono applicati: sono filtri di sistema a valle che richiedono
  l'associazione alla persona, non qualita' del detector. Qui si misura il detector.
- Cigarette: SH17 non la annota -> non calcolabile P/R. Si riporta solo il numero
  di detection "Cigarette" come proxy di falsi positivi (nessuna sigaretta attesa
  in un dataset di sicurezza industriale).
"""
import json
import sys
import time
from pathlib import Path

import cv2 as cv

sys.path.insert(0, "src")
from visual_security.analyzer import build_detector  # noqa: E402
from visual_security.person_ppe_checker import _iou  # noqa: E402

SUBSET = Path("evaluation/sh17_subset")
EVAL_CLASSES = ["Person", "Helmet", "Vest", "Glasses", "Glove", "Shoe"]
IOU_THR = 0.5
DETECTORS = ["grounding-dino", "omdet-turbo"]

gt_data = json.load(open(SUBSET / "ground_truth.json", encoding="utf-8"))
print(f"Immagini nel subset: {len(gt_data)}")


def match(preds, gts):
    """preds: list (conf, xyxy) confidence-desc; gts: list xyxy. Ritorna (tp, fp, fn)."""
    used = [False] * len(gts)
    tp = 0
    for _, pb in sorted(preds, key=lambda p: -p[0]):
        best_i, best_iou = -1, IOU_THR
        for i, gb in enumerate(gts):
            if used[i]:
                continue
            iou = _iou(tuple(pb), tuple(gb))
            if iou >= best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0:
            used[best_i] = True
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


results = {}
for name in DETECTORS:
    print(f"\n{'=' * 60}\n[{name}] valutazione su {len(gt_data)} immagini\n{'=' * 60}", flush=True)
    det = build_detector(name)
    # Warmup: la prima inferenza include il caricamento pigro del modello
    # (~decine di secondi); la si esclude dalle statistiche di latenza.
    _warm = cv.imread(str(SUBSET / "images" / gt_data[0]["file"]))
    det.analyze(_warm)
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in EVAL_CLASSES}
    cig_fp = 0
    times = []
    t_start = time.perf_counter()
    for idx, item in enumerate(gt_data):
        img = cv.imread(str(SUBSET / "images" / item["file"]))
        if img is None:
            continue
        res = det.analyze(img)
        times.append(res.inference_time_ms)
        if res.error:
            print(f"  [{item['file']}] ERROR {res.error}", flush=True)
            continue
        # predizioni per classe
        preds = {c: [] for c in EVAL_CLASSES}
        for d in res.detections:
            if d.label in preds:
                preds[d.label].append((d.confidence, d.bbox))
            elif d.label == "Cigarette":
                cig_fp += 1
        # GT per classe
        gts = {c: [b[1:] for b in item["boxes"] if b[0] == c] for c in EVAL_CLASSES}
        for c in EVAL_CLASSES:
            tp, fp, fn = match(preds[c], gts[c])
            counts[c]["tp"] += tp
            counts[c]["fp"] += fp
            counts[c]["fn"] += fn
        if (idx + 1) % 40 == 0:
            print(f"  {idx + 1}/{len(gt_data)}  ({sum(times) / len(times):.0f} ms/img medi)", flush=True)
    wall = time.perf_counter() - t_start

    # metriche
    per_class = {}
    for c in EVAL_CLASSES:
        tp, fp, fn = counts[c]["tp"], counts[c]["fp"], counts[c]["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3),
                        "recall": round(rec, 3), "f1": round(f1, 3)}
    macro_f1 = round(sum(v["f1"] for v in per_class.values()) / len(per_class), 3)
    macro_p = round(sum(v["precision"] for v in per_class.values()) / len(per_class), 3)
    macro_r = round(sum(v["recall"] for v in per_class.values()) / len(per_class), 3)
    ttp = sum(v["tp"] for v in per_class.values())
    tfp = sum(v["fp"] for v in per_class.values())
    tfn = sum(v["fn"] for v in per_class.values())
    micro_p = round(ttp / (ttp + tfp), 3) if (ttp + tfp) else 0.0
    micro_r = round(ttp / (ttp + tfn), 3) if (ttp + tfn) else 0.0
    micro_f1 = round(2 * micro_p * micro_r / (micro_p + micro_r), 3) if (micro_p + micro_r) else 0.0

    results[name] = {
        "model_id": det.model_id,
        "n_images": len(gt_data),
        "iou_threshold": IOU_THR,
        "per_class": per_class,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "cigarette_detections_proxy_fp": cig_fp,
        "latency_ms": {
            "mean": round(sum(times) / len(times), 1),
            "min": round(min(times), 1),
            "max": round(max(times), 1),
        },
        "wall_seconds": round(wall, 1),
    }
    print(f"[{name}] macro-F1={macro_f1} micro-F1={micro_f1} "
          f"lat={results[name]['latency_ms']['mean']}ms cig_fp={cig_fp} wall={wall:.0f}s", flush=True)
    for c in EVAL_CLASSES:
        v = per_class[c]
        print(f"    {c:9} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} "
              f"(tp={v['tp']} fp={v['fp']} fn={v['fn']})", flush=True)

json.dump(results, open(SUBSET.parent / "eval_static_results.json", "w", encoding="utf-8"), indent=2)
print("\nRisultati salvati in evaluation/eval_static_results.json")
