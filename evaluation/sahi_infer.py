"""
SAHI minimale (Slicing Aided Hyper Inference): esegue il detector sull'immagine
intera + su tile sovrapposte, riporta i box nelle coordinate originali e fonde
tutto con NMS per-classe. Aiuta gli oggetti piccoli (guanti/scarpe/occhiali)
perche' su un tile occupano piu' pixel. Costo: N inferenze per immagine.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")
from visual_security.analyzer import Detection, nms_per_class  # noqa: E402


def sahi_detect(detector, img, tile_frac=0.6, overlap=0.25, merge_iou=0.5, include_full=True):
    """Ritorna una lista di Detection fuse (coordinate immagine originale)."""
    h, w = img.shape[:2]
    tw, th = int(w * tile_frac), int(h * tile_frac)
    step_x = max(1, int(tw * (1 - overlap)))
    step_y = max(1, int(th * (1 - overlap)))

    def coords(size, tile, step):
        xs = list(range(0, max(1, size - tile + 1), step)) or [0]
        if xs[-1] + tile < size:
            xs.append(size - tile)
        return xs

    dets: list[Detection] = []
    if include_full:
        dets.extend(detector.analyze(img).detections)
    for y in coords(h, th, step_y):
        for x in coords(w, tw, step_x):
            res = detector.analyze(img[y:y + th, x:x + tw])
            for d in res.detections:
                if d.bbox is None:
                    continue
                b = d.bbox
                dets.append(Detection(d.label, d.confidence, [b[0] + x, b[1] + y, b[2] + x, b[3] + y]))
    return nms_per_class(dets, merge_iou)
