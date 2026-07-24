"""
Metriche OFFLINE dai dump di detection grezze (eval_capture.py). Nessun modello
rieseguito: si ricalcola tutto variando soglia di confidence e IoU.

Per ogni detector e classe:
- AP (average precision, area sotto la curva P-R, interpolazione VOC all-points)
  a IoU 0.5 e 0.3 -> indipendente dalla soglia, misura la qualita' "intrinseca".
- F1 al PUNTO OPERATIVO di produzione (soglie dei video: base 0.35 GD / 0.30 OmDet,
  Glasses 0.45).
- F1 MASSIMO ottenibile e la soglia che lo realizza (sweep) -> quanto recall
  lasciamo sul tavolo con le soglie alte.
- Cigarette: nessun GT -> solo conteggio detection >= 0.50 (proxy FP).
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, "src")
from visual_security.person_ppe_checker import _iou  # noqa: E402

EVAL = ["Person", "Helmet", "Vest", "Glasses", "Glove", "Shoe"]
DETECTORS = ["grounding-dino", "omdet-turbo"]
BASE_THR = {"grounding-dino": 0.35, "omdet-turbo": 0.30}
PERCLASS_THR = {"Glasses": 0.45, "Cigarette": 0.50}
IOUS = [0.5, 0.3]
SWEEP = [i / 100 for i in range(5, 96, 5)]

SUB = Path("evaluation/sh17_subset")
gt_data = json.load(open(SUB / "ground_truth.json", encoding="utf-8"))


def gt_by_image(cls):
    return {g["file"]: [b[1:] for b in g["boxes"] if b[0] == cls] for g in gt_data}


def nms_raw(dets, iou_thr=0.5):
    """NMS per-classe sulle detection grezze [label,x1,y1,x2,y2,conf]."""
    from collections import defaultdict

    by_label = defaultdict(list)
    for d in dets:
        by_label[d[0]].append(d)
    kept = []
    for group in by_label.values():
        for d in sorted(group, key=lambda x: -x[5]):
            if all(_iou(tuple(d[1:5]), tuple(k[1:5])) <= iou_thr for k in kept if k[0] == d[0]):
                kept.append(d)
    return kept


def voc_ap(rec, prec):
    mrec = [0.0, *rec, 1.0]
    mpre = [0.0, *prec, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i] - mrec[i - 1]) * mpre[i] for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1])


def eval_class(preds_by_img, gt_img, iou_thr):
    """preds_by_img: {file:[(conf,bbox)]}. Ritorna (ap, confs, tp_flags, total_gt)."""
    total_gt = sum(len(v) for v in gt_img.values())
    allp = sorted(((c, f, b) for f, lst in preds_by_img.items() for (c, b) in lst), key=lambda x: -x[0])
    matched = {f: [False] * len(v) for f, v in gt_img.items()}
    tp_cum = fp_cum = 0
    precisions, recalls, confs, tps = [], [], [], []
    for conf, f, b in allp:
        best_iou, best_j = iou_thr, -1
        for j, gb in enumerate(gt_img.get(f, [])):
            if matched[f][j]:
                continue
            iou = _iou(tuple(b), tuple(gb))
            if iou >= best_iou:
                best_iou, best_j = iou, j
        is_tp = best_j >= 0
        if is_tp:
            matched[f][best_j] = True
            tp_cum += 1
        else:
            fp_cum += 1
        confs.append(conf)
        tps.append(is_tp)
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt if total_gt else 0.0)
    ap = voc_ap(recalls, precisions) if total_gt else 0.0
    return ap, confs, tps, total_gt


def prf_at(confs, tps, total_gt, t):
    tp = sum(1 for c, x in zip(confs, tps) if c >= t and x)
    fp = sum(1 for c, x in zip(confs, tps) if c >= t and not x)
    fn = total_gt - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 3), round(r, 3), round(f1, 3), tp, fp, fn


def compute(name, raw):
    per_class = {}
    cig_all = sum(1 for v in raw.values() for d in v if d[0] == "Cigarette")
    cig_prod = sum(1 for v in raw.values() for d in v if d[0] == "Cigarette" and d[5] >= PERCLASS_THR["Cigarette"])
    for cls in EVAL:
        gt_img = gt_by_image(cls)
        preds_by_img = {f: [(d[5], d[1:5]) for d in dets if d[0] == cls] for f, dets in raw.items()}
        ap50, confs, tps, tot = eval_class(preds_by_img, gt_img, 0.5)
        ap30, _, _, _ = eval_class(preds_by_img, gt_img, 0.3)
        pth = PERCLASS_THR.get(cls, BASE_THR[name])
        p, r, f1, tp, fp, fn = prf_at(confs, tps, tot, pth)
        best = max(((prf_at(confs, tps, tot, t), t) for t in SWEEP), key=lambda z: z[0][2])
        (bp, br, bf1, *_), bt = best
        per_class[cls] = {
            "ap50": round(ap50, 3), "ap30": round(ap30, 3), "n_gt": tot,
            "prod": {"threshold": pth, "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
            "best_f1": {"threshold": bt, "precision": bp, "recall": br, "f1": bf1},
        }
    macro = lambda k: round(sum(per_class[c][k] for c in EVAL) / len(EVAL), 3)  # noqa: E731
    return {
        "map50": macro("ap50"),
        "map30": macro("ap30"),
        "macro_f1_prod": round(sum(per_class[c]["prod"]["f1"] for c in EVAL) / len(EVAL), 3),
        "macro_f1_best": round(sum(per_class[c]["best_f1"]["f1"] for c in EVAL) / len(EVAL), 3),
        "cigarette_det_total": cig_all,
        "cigarette_det_at_prod_thr": cig_prod,
        "per_class": per_class,
    }


# Calcola ogni detector in due varianti: detection grezze e con NMS per-classe
# (IoU 0.5) applicata offline sugli stessi dump -> mostra l'effetto della NMS
# senza rieseguire i modelli.
results = {}
for name in DETECTORS:
    raw0 = json.load(open(SUB.parent / f"raw_detections_{name}.json"))
    results[name] = {
        "no_nms": compute(name, raw0),
        "nms": compute(name, {f: nms_raw(dets, 0.5) for f, dets in raw0.items()}),
    }

json.dump(results, open("evaluation/eval_metrics_results.json", "w"), indent=2)

# ── stampa leggibile ────────────────────────────────────────────────────────
for name in DETECTORS:
    for variant in ["no_nms", "nms"]:
        R = results[name][variant]
        tag = "senza NMS" if variant == "no_nms" else "con NMS  "
        print(f"\n{'=' * 78}\n{name} [{tag}]  |  mAP@0.5={R['map50']}  mAP@0.3={R['map30']}  "
              f"macroF1(prod)={R['macro_f1_prod']}\n{'=' * 78}")
        print(f"{'classe':9} {'AP@.5':>6} {'AP@.3':>6} | {'F1 prod':>8} (P/R)")
        for c in EVAL:
            v = R["per_class"][c]; pr = v["prod"]
            print(f"{c:9} {v['ap50']:>6.3f} {v['ap30']:>6.3f} | "
                  f"{pr['f1']:>8.2f} ({pr['precision']:.2f}/{pr['recall']:.2f})")
        print(f"Cigarette (no GT): {R['cigarette_det_total']} det, "
              f"{R['cigarette_det_at_prod_thr']} a soglia >= {PERCLASS_THR['Cigarette']}")

print(f"\n{'=' * 78}\nEFFETTO NMS (delta mAP@.5 / macroF1 prod)\n{'=' * 78}")
for name in DETECTORS:
    a, b = results[name]["no_nms"], results[name]["nms"]
    print(f"{name:16} mAP@.5 {a['map50']:.3f} -> {b['map50']:.3f} ({b['map50'] - a['map50']:+.3f})   "
          f"macroF1 {a['macro_f1_prod']:.3f} -> {b['macro_f1_prod']:.3f} ({b['macro_f1_prod'] - a['macro_f1_prod']:+.3f})")
print("\nSalvato evaluation/eval_metrics_results.json")
