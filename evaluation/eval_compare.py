"""
Confronto offline di TUTTE le varianti (baseline + esperimenti Tier 1/2) sullo
stesso subset SH17, riusando le funzioni di eval_metrics. Applica la NMS
per-classe a tutte (coerente con la pipeline attuale). Nessun modello rieseguito.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import BASE_THR, compute, nms_raw  # noqa: E402

SUB = Path("evaluation/sh17_subset")
GD, OM, OW = BASE_THR["grounding-dino"], BASE_THR["omdet-turbo"], BASE_THR["owlv2"]

# etichetta -> (file dump, soglia base del detector sottostante)
VARIANTS = {
    "GD baseline": ("raw_detections_grounding-dino.json", GD),
    "GD multi-synonym": ("raw_detections_gd_multisyn.json", GD),
    "OmDet baseline": ("raw_detections_omdet-turbo.json", OM),
    "OmDet multi-synonym": ("raw_detections_omdet_multisyn.json", OM),
    "OmDet + SAHI": ("raw_detections_omdet_sahi.json", OM),
    "OWLv2": ("raw_detections_owlv2.json", OW),
}

results = {}
for name, (fn, base) in VARIANTS.items():
    p = SUB.parent / fn
    if not p.exists():
        print(f"[skip] {name}: {fn} assente")
        continue
    raw = json.load(open(p))
    raw_nms = {f: nms_raw(dets, 0.5) for f, dets in raw.items()}
    results[name] = compute(raw_nms, base)

# Ensemble sintetico: OWLv2 per tutte le classi tranne Glasses, GD per Glasses
# (OWLv2 e' debole sugli occhiali, GD e' il migliore). AP e' cio' che conta qui;
# la F1 "prod" e' mista quindi solo indicativa.
owl_p, gd_p = SUB.parent / "raw_detections_owlv2.json", SUB.parent / "raw_detections_grounding-dino.json"
if owl_p.exists() and gd_p.exists():
    owl_d, gd_d = json.load(open(owl_p)), json.load(open(gd_p))
    ens = {
        f: nms_raw([d for d in owl_d[f] if d[0] != "Glasses"] + [d for d in gd_d.get(f, []) if d[0] == "Glasses"], 0.5)
        for f in owl_d
    }
    results["Ensemble OWLv2+GD(glasses)"] = compute(ens, OW)

json.dump(results, open("evaluation/eval_experiments_results.json", "w"), indent=2)

print(f"\n{'variante':22} {'mAP@.5':>7} {'mAP@.3':>7} {'macroF1':>8}")
print("-" * 48)
for name, r in results.items():
    print(f"{name:22} {r['map50']:>7.3f} {r['map30']:>7.3f} {r['macro_f1_prod']:>8.3f}")

order = ["Person", "Helmet", "Vest", "Glasses", "Glove", "Shoe"]
print(f"\nAP@.5 per classe:\n{'variante':22} " + " ".join(f"{c[:6]:>6}" for c in order))
print("-" * (22 + 7 * len(order)))
for name, r in results.items():
    print(f"{name:22} " + " ".join(f"{r['per_class'][c]['ap50']:>6.3f}" for c in order))
print("\nSalvato evaluation/eval_experiments_results.json")
