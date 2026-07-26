"""
Cattura le detection grezze per gli esperimenti di miglioramento metriche
(Tier 1 + Tier 2), sullo stesso subset SH17. Ogni variante -> un dump
raw_detections_<variant>.json, poi valutato offline da eval_compare.py.

Varianti:
  omdet_multisyn : OmDet-Turbo con prompt MULTI-SINONIMO (piu' frasi per classe)
  gd_multisyn    : Grounding DINO con prompt MULTI-SINONIMO
  omdet_sahi     : OmDet-Turbo con SAHI (inferenza a tile per oggetti piccoli)
  owlv2          : OWLv2 (backend alternativo, Apache 2.0) con prompt base

Tutte a soglia bassa (0.05) e filtro per-classe OFF: il threshold/NMS si applica
poi offline, coerente con la cattura baseline (eval_capture.py).
"""
import json
import sys
import time
from pathlib import Path

import cv2 as cv

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import visual_security.analyzer as az  # noqa: E402
from visual_security.analyzer import GroundingDinoAnalyzer, OmDetTurboAnalyzer, Owlv2Analyzer  # noqa: E402
from sahi_infer import sahi_detect  # noqa: E402

SUBSET = Path("evaluation/sh17_subset")
KEEP = {"Person", "Helmet", "Vest", "Glasses", "Glove", "Shoe", "Cigarette"}
gt = json.load(open(SUBSET / "ground_truth.json", encoding="utf-8"))
SMOKE = "--smoke" in sys.argv
items = gt[:3] if SMOKE else gt
print(f"Immagini: {len(items)}{' (SMOKE)' if SMOKE else ''}")

# Prompt multi-sinonimo: piu' formulazioni per classe (nomi distintivi per non
# confondere il matching di Grounding DINO). Piu' categorie nel prompt = piu'
# oggetti rilevati (letteratura: puo' alzare parecchio l'mAP).
MULTI_SYNONYM_PROMPTS = {
    "a person": "Person",
    "a worker": "Person",
    "a hard hat": "Helmet",
    "a construction helmet": "Helmet",
    "a reflective safety vest": "Vest",
    "a high-visibility vest": "Vest",
    "safety glasses": "Glasses",
    "protective eyewear": "Glasses",
    "a work glove": "Glove",
    "protective gloves": "Glove",
    "a work boot": "Shoe",
    "protective footwear": "Shoe",
    "a cigarette": "Cigarette",
}
# Keyword extra per il matching di Grounding DINO sui sotto-frasi delle sinonimie
# (monkeypatch locale: NON tocca la produzione).
az._KEYWORD_TO_LABEL = az._KEYWORD_TO_LABEL + [
    ("worker", "Person"), ("high-visibility", "Vest"), ("hi-vis", "Vest"),
    ("eyewear", "Glasses"), ("footwear", "Shoe"),
]


def to_rows(dets):
    return [
        [d.label, round(d.bbox[0], 1), round(d.bbox[1], 1), round(d.bbox[2], 1), round(d.bbox[3], 1),
         round(float(d.confidence), 4)]
        for d in dets if d.label in KEEP
    ]


def capture(variant, infer):
    print(f"\n[{variant}] cattura...", flush=True)
    infer(cv.imread(str(SUBSET / "images" / items[0]["file"])))  # warmup
    out, t0 = {}, time.perf_counter()
    for i, it in enumerate(items):
        img = cv.imread(str(SUBSET / "images" / it["file"]))
        out[it["file"]] = to_rows(infer(img))
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(items)}  ({(time.perf_counter() - t0) / (i + 1):.1f}s/img)", flush=True)
    if not SMOKE:
        json.dump(out, open(f"evaluation/raw_detections_{variant}.json", "w"), separators=(",", ":"))
    n = sum(len(v) for v in out.values())
    print(f"[{variant}] {n} detection in {time.perf_counter() - t0:.0f}s", flush=True)
    return out


# ── Costruzione detector (soglia bassa, filtro per-classe OFF) ──────────────
def mk(cls, **kw):
    d = cls(conf_threshold=0.05, **kw)
    d.class_conf = {}
    return d


omdet_ms = mk(OmDetTurboAnalyzer, prompts=MULTI_SYNONYM_PROMPTS)
gd_ms = mk(GroundingDinoAnalyzer, text_threshold=0.10, prompts=MULTI_SYNONYM_PROMPTS)
omdet_base = mk(OmDetTurboAnalyzer)
owlv2 = mk(Owlv2Analyzer)

# Ordine per costo crescente (le varianti veloci finiscono prima)
capture("omdet_multisyn", lambda img: omdet_ms.analyze(img).detections)
capture("owlv2", lambda img: owlv2.analyze(img).detections)
capture("omdet_sahi", lambda img: sahi_detect(omdet_base, img, tile_frac=0.6, overlap=0.25))
capture("gd_multisyn", lambda img: gd_ms.analyze(img).detections)

print("\nCattura esperimenti completata.")
