"""
Costruisce un subset bilanciato di SH17 per la valutazione statica dei detector.

SH17 (fathansanum/SH-17-Dataset, COCO, CC BY 4.0, 8099 img, annotazioni COMPLETE)
e' il dataset PPE piu' completo trovato: copre 6 delle nostre 7 classi
(manca solo Cigarette, che nessun dataset PPE ha).

Mappatura category_id COCO -> nostra classe, VERIFICATA VISIVAMENTE (l'export
Roboflow ha rinominato le classi con indici numerici NON allineati all'ordine
ufficiale SH17, quindi la mappa e' stata ricavata ispezionando i crop):
    1 -> Person, 3 -> Helmet, 9 -> Vest, 16 -> Glasses, 17 -> Glove, 7 -> Shoe

Campionamento bilanciato: quote per-classe sulle classi rare (union+dedup),
Person viene incluso di conseguenza (presente nel ~94% delle immagini).
Scarica SOLO le immagini selezionate (non i ~4GB completi).
"""
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "fathansanum/SH-17-Dataset"
META = Path("evaluation/sh17_meta")
OUT = Path("evaluation/sh17_subset")
OUT_IMG = OUT / "images"
OUT_IMG.mkdir(parents=True, exist_ok=True)

CAT2OURS = {1: "Person", 3: "Helmet", 9: "Vest", 16: "Glasses", 17: "Glove", 7: "Shoe"}
# quote per-classe (immagini che contengono la classe); Person non ha quota
QUOTA = {"Helmet": 100, "Vest": 100, "Glasses": 100, "Glove": 100, "Shoe": 100}
SEED = 42

# ── Carica tutte le annotazioni, indicizza per immagine ─────────────────────
# img_key = (split, image_id) -> {"file": file_name, "boxes": [(cls, x1,y1,x2,y2)], "classes": set}
images = {}
for split in ["valid", "test", "train"]:
    d = json.load(open(META / f"SH17-dataset-1/{split}/_annotations.coco.json", encoding="utf-8"))
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    for im in d["images"]:
        images[(split, im["id"])] = {"split": split, "file": im["file_name"], "boxes": [], "classes": set()}
    for a in d["annotations"]:
        c = a["category_id"]
        if c not in CAT2OURS:
            continue
        cls = CAT2OURS[c]
        x, y, w, h = a["bbox"]
        key = (split, a["image_id"])
        images[key]["boxes"].append([cls, x, y, x + w, y + h])
        images[key]["classes"].add(cls)

OUR_CLASSES = ["Person", "Helmet", "Vest", "Glasses", "Glove", "Shoe"]
by_class = {cls: [k for k, v in images.items() if cls in v["classes"]] for cls in OUR_CLASSES}

# ── Selezione bilanciata: classi rare prima ─────────────────────────────────
rnd = random.Random(SEED)
selected = set()
for cls in sorted(QUOTA, key=lambda c: len(by_class[c])):  # rarest first
    pool = by_class[cls][:]
    rnd.shuffle(pool)
    have = sum(1 for k in selected if cls in images[k]["classes"])
    for k in pool:
        if have >= QUOTA[cls]:
            break
        if k not in selected:
            selected.add(k)
            have += 1

print(f"immagini selezionate: {len(selected)}")

# ── Scarica le immagini selezionate + costruisci GT ─────────────────────────
gt = []
from collections import Counter

inst = Counter()
for i, key in enumerate(sorted(selected)):
    v = images[key]
    src = f"SH17-dataset-1/{v['split']}/{v['file']}"
    local = hf_hub_download(REPO, src, repo_type="dataset", local_dir=str(META))
    # nome univoco per evitare collisioni tra split
    dst_name = f"{v['split']}_{v['file']}"
    dst = OUT_IMG / dst_name
    if not dst.exists():
        dst.write_bytes(Path(local).read_bytes())
    for b in v["boxes"]:
        inst[b[0]] += 1
    gt.append({"file": dst_name, "boxes": v["boxes"]})
    if (i + 1) % 40 == 0:
        print(f"  scaricate {i + 1}/{len(selected)}")

json.dump(gt, open(OUT / "ground_truth.json", "w", encoding="utf-8"), indent=1)
print("GT salvato in", OUT / "ground_truth.json")
print("istanze per classe nel subset:", dict(inst))
print("immagini con ogni classe:")
for cls in OUR_CLASSES:
    n = sum(1 for g in gt if any(b[0] == cls for b in g["boxes"]))
    print(f"  {cls:10} {n:>4} img  {inst[cls]:>5} istanze")
