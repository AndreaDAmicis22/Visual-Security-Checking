"""
debug_frame.py — Diagnose PPE detection on a single image or video frame.

Usage:
    python -m src.visual_security.debug_frame \\
        --image path/to/frame.jpg \\
        --yolo-model weights/best.onnx

Prints raw YOLO detections + bboxes, then runs PersonPPEChecker and shows
the full per-person association so you can tune thresholds without running
the full video loop.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np


def run(image_path: str, yolo_model: str, conf: float, cont_thr: float, iou_thr: float):
    from .analyzer import YOLOAnalyzer
    from .person_ppe_checker import PersonPPEChecker, _to_xyxy

    print(f"\n{'='*60}")
    print(f"DEBUG FRAME: {image_path}")
    print(f"Model:  {yolo_model}   conf≥{conf}")
    print(f"Containment≥{cont_thr}   IoU≥{iou_thr}")
    print('='*60)

    yolo = YOLOAnalyzer(model_path=yolo_model, conf_threshold=conf)
    img  = cv.imread(image_path)
    if img is None:
        print(f"[ERRORE] Impossibile leggere: {image_path}")
        return

    h, w = img.shape[:2]
    result = yolo.analyze(img)   # ndarray diretto — niente disco

    if result.error:
        print(f"[ERROR] YOLO failed: {result.error}")
        return

    print(f"\nYOLO — {len(result.detections)} detections  ({result.inference_time_ms:.1f} ms)")
    for i, d in enumerate(result.detections):
        box = _to_xyxy(d.bbox)
        box_str = f"[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]" if box else "NO_BBOX"
        print(f"  [{i:2d}] {d.label:<18} conf={d.confidence:.3f}  bbox={box_str}")

    checker = PersonPPEChecker(
        containment_threshold=cont_thr,
        iou_threshold=iou_thr,
    )
    persons = checker.check(result.detections, frame_w=w, frame_h=h)

    print(f"\nPersonPPEChecker — {len(persons)} person(s) found")
    for pr in persons:
        print(f"\n  {pr.summary()}")
        if pr.associated_ppe:
            for det in pr.associated_ppe:
                box = _to_xyxy(det.bbox)
                box_str = f"[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]" if box else "?"
                print(f"    ↳ {det.label:<14} conf={det.confidence:.2f}  bbox={box_str}")

    # Draw and save annotated image
    img = cv.imread(image_path)
    if img is not None:
        for pr in persons:
            if pr.person_bbox:
                x1,y1,x2,y2 = (int(v) for v in pr.person_bbox)
                color = (40,200,60) if pr.is_compliant else (0,50,220)
                cv.rectangle(img,(x1,y1),(x2,y2),color,2)
                label = "OK" if pr.is_compliant else f"MISSING:{','.join(pr.missing_ppe)}"
                cv.putText(img, label, (x1, max(y1-6,14)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cy = y1+18
                for cat, req in PersonPPEChecker.FULL_PPE.items():
                    found = pr.found_ppe.get(cat,0)
                    ok = found >= req
                    c = (40,200,60) if ok else (0,50,220)
                    cv.putText(img, f"{'v' if ok else 'x'}{cat}({found}/{req})",
                               (x1+4, cy), cv.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
                    cy += 15
        for det in result.detections:
            if det.label.lower() != "person":
                box = _to_xyxy(det.bbox)
                if box:
                    cv.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(200,200,50),1)
                    cv.putText(img,det.label,(int(box[0]),int(box[1])-4),
                               cv.FONT_HERSHEY_SIMPLEX,0.35,(200,200,50),1)
        out = Path(image_path).stem + "_debug.jpg"
        cv.imwrite(out, img)
        print(f"\nAnnotated image saved → {out}")

    print()


def main():
    p = argparse.ArgumentParser(description="Debug single-frame PPE detection")
    p.add_argument("--image",       required=True)
    p.add_argument("--yolo-model",  required=True)
    p.add_argument("--conf",        type=float, default=0.35)
    p.add_argument("--containment", type=float, default=0.30,
                   help="Containment threshold for PPE→Person association")
    p.add_argument("--iou",         type=float, default=0.05)
    args = p.parse_args()
    run(args.image, args.yolo_model, args.conf, args.containment, args.iou)


if __name__ == "__main__":
    main()
