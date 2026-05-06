"""
debug_video.py — Diagnostica le raw detection di YOLO su N frame campionati dal video.
Mostra il formato bbox restituito dal backend, senza scrivere nulla su disco.

Usage:
    python -m src.visual_security.debug_video \
        --video test.mp4 \
        --yolo-model weights/dataset_1/yolo_nano_640/best.onnx \
        --samples 6
"""

from __future__ import annotations

import argparse
import sys


def run(video: str, yolo_model: str, conf: float, samples: int) -> None:
    import cv2 as cv

    sys.path.insert(0, "src")
    from visual_security.analyzer import YOLOAnalyzer
    from visual_security.person_ppe_checker import _to_xyxy

    yolo = YOLOAnalyzer(model_path=yolo_model, conf_threshold=conf)

    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire: {video}")
        return

    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo : {video}")
    print(f"Frame : {n_frames}  @  {fps:.1f} fps  |  {fw}×{fh} px")
    print(f"Modello: {yolo_model}   conf≥{conf}")
    print("=" * 70)

    step = max(1, n_frames // samples)

    for i in range(samples):
        fi = i * step
        cap.set(cv.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f"\nFrame {fi:5d}: lettura fallita, skip.")
            continue

        res = yolo.analyze(frame)

        ts = fi / fps
        print(f"\nFrame {fi:5d}  ({ts:.1f}s)  →  {len(res.detections)} det  [{res.inference_time_ms:.0f} ms]")

        if res.error:
            print(f"  [ERRORE] {res.error}")
            continue

        if not res.detections:
            print("  (nessuna detection)")
            continue

        for d in res.detections:
            raw = d.bbox
            parsed = _to_xyxy(raw, fw, fh)

            if parsed:
                bw = parsed[2] - parsed[0]
                bh = parsed[3] - parsed[1]
                size_str = f"  size={bw:.0f}×{bh:.0f}px"

                # Avvisi formato
                if raw and len(raw) == 4 and all(isinstance(v, float) for v in raw):
                    if all(0.0 <= v <= 1.0 for v in raw):
                        size_str += "  ⚠ NORMALIZZATO (0-1)"
                    raw3, raw2 = raw[2], raw[0]
                    if raw3 < raw2:
                        size_str += "  ⚠ PROB. CX,CY,W,H"
            else:
                size_str = "  bbox=NON_PARSEABLE"

            print(f"  {d.label:<18} conf={d.confidence:.3f}  raw_bbox={raw}{size_str}")

    cap.release()
    print("\n" + "=" * 70)

    # ── Suggerimenti ──────────────────────────────────────────────────────────
    print("\nSuggerimento: se tutte le bbox hanno valori 0-1, il backend restituisce")
    print("coordinate normalizzate → il checker le converte automaticamente.")
    print("Se non vedi 'Person' nelle detection, controlla DEFAULT_CLASS_NAMES in analyzer.py.")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnostica detection YOLO su frame campionati dal video (no file su disco)")
    p.add_argument("--video", required=True, help="Percorso del video")
    p.add_argument("--yolo-model", required=True, help="Percorso modello ONNX")
    p.add_argument("--conf", type=float, default=0.25, help="Soglia confidence (default 0.25)")
    p.add_argument("--samples", type=int, default=6, help="Numero di frame da campionare")
    args = p.parse_args()
    run(args.video, args.yolo_model, args.conf, args.samples)


if __name__ == "__main__":
    main()
