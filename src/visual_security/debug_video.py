"""
debug_video.py - Diagnostica le raw detection su N frame campionati dal video.
Mostra il formato bbox restituito dal backend, senza scrivere nulla su disco.

Usage:
    python -m src.visual_security.debug_video \
        --video test.mp4 \
        --detector grounding-dino \
        --samples 6
"""

from __future__ import annotations

import argparse
import sys


def run(video: str, detector_name: str, conf: float | None, samples: int) -> None:
    import cv2 as cv

    sys.path.insert(0, "src")
    from visual_security.analyzer import build_detector
    from visual_security.person_ppe_checker import _to_xyxy

    det = build_detector(detector_name, conf_threshold=conf)

    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire: {video}")
        return

    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo : {video}")
    print(f"Frame : {n_frames}  @  {fps:.1f} fps  |  {fw}x{fh} px")
    print(f"Detector: {detector_name}   conf>={conf if conf is not None else 'default'}")
    print("=" * 70)

    step = max(1, n_frames // samples)

    for i in range(samples):
        fi = i * step
        cap.set(cv.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f"\nFrame {fi:5d}: lettura fallita, skip.")
            continue

        res = det.analyze(frame)

        ts = fi / fps
        print(f"\nFrame {fi:5d}  ({ts:.1f}s)  ->  {len(res.detections)} det  [{res.inference_time_ms:.0f} ms]")

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
                size_str = f"  size={bw:.0f}x{bh:.0f}px"

                # Avvisi formato
                if raw and len(raw) == 4 and all(isinstance(v, float) for v in raw):
                    if all(0.0 <= v <= 1.0 for v in raw):
                        size_str += "  ! NORMALIZZATO (0-1)"
                    raw3, raw2 = raw[2], raw[0]
                    if raw3 < raw2:
                        size_str += "  ! PROB. CX,CY,W,H"
            else:
                size_str = "  bbox=NON_PARSEABLE"

            print(f"  {d.label:<18} conf={d.confidence:.3f}  raw_bbox={raw}{size_str}")

    cap.release()
    print("\n" + "=" * 70)

    # ── Suggerimenti ──────────────────────────────────────────────────────────
    print("\nSuggerimento: se tutte le bbox hanno valori 0-1, il backend restituisce")
    print("coordinate normalizzate -> il checker le converte automaticamente.")
    print("Se non vedi 'Person' nelle detection, controlla DETECTION_PROMPTS in analyzer.py.")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnostica detection su frame campionati dal video (no file su disco)")
    p.add_argument("--video", required=True, help="Percorso del video")
    p.add_argument("--detector", default="grounding-dino", choices=["grounding-dino", "omdet-turbo"])
    p.add_argument("--conf", type=float, default=None, help="Soglia confidence (default: backend default)")
    p.add_argument("--samples", type=int, default=6, help="Numero di frame da campionare")
    args = p.parse_args()
    run(args.video, args.detector, args.conf, args.samples)


if __name__ == "__main__":
    main()
