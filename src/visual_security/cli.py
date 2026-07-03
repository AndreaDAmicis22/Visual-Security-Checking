"""
CLI for Visual Security — PPE Tracker.

Usage examples:

  # Track a video file (YOLO only, no VLM)
  python -m visual_security.cli track \
      --yolo-model weights/dataset_1/yolo_nano_640/best.onnx \
      --source video.mp4 \
      --no-vlm

  # Track with local VLM escalation (SmolVLM, in-process — no server)
  python -m visual_security.cli track \
      --yolo-model weights/best.onnx \
      --source video.mp4 \
      --vlm-model HuggingFaceTB/SmolVLM-500M-Instruct \
      --save-output output/annotated.mp4 \
      --alert-log output/alerts.json

  # Track from webcam
  python -m visual_security.cli track \
      --yolo-model weights/best.onnx \
      --source 0

  # Check if the VLM backend (torch/transformers) is available
  python -m visual_security.cli check-vlm
"""

from __future__ import annotations

import argparse


def cmd_track(args):
    """Real-time video tracking with YOLO + optional local VLM escalation."""
    from .video_tracker import build_tracker

    tracker = build_tracker(
        yolo_model_path=args.yolo_model,
        vlm_model="none" if args.no_vlm else args.vlm_model,
        persistence_frames=args.persistence,
        window_frames=args.window,
        skip_frames=args.skip_frames,
        display=not args.no_display,
        save_output=args.save_output,
        alert_log=args.alert_log,
        yolo_conf=args.conf,
        verbose=args.verbose,
    )

    source = int(args.source) if args.source.isdigit() else args.source
    alerts = tracker.run(source)

    print(f"\n{'=' * 60}")
    print(f"Tracking complete — {len(alerts)} confirmed alert(s).")
    print(f"{'=' * 60}")
    for a in alerts:
        print(a.summary())


def cmd_check_vlm(args):
    """Verify that the local VLM backend (torch/transformers) is importable."""
    from .vlm_validator import LocalVLMValidator

    if LocalVLMValidator.is_available():
        print("OK: torch + transformers disponibili. Il VLM locale può essere usato.")
        print(f"    Modello di default: '{args.model}' (scaricato al primo utilizzo).")
    else:
        print("ERROR: torch/transformers non installati.")
        print("\nSetup:")
        print("  pip install torch transformers pillow einops accelerate")


def main():
    parser = argparse.ArgumentParser(
        prog="visual_security",
        description="PPE Safety Tracker — YOLO + VLM escalation",
    )
    sub = parser.add_subparsers(dest="command")

    # ── track ─────────────────────────────────────────────────────────────────
    p_track = sub.add_parser("track", help="Run real-time PPE tracking on video/webcam")
    p_track.add_argument("--yolo-model", required=True, help="Path to YOLO ONNX model")
    p_track.add_argument("--source", default="0", help="Video file path or camera index (default: 0)")
    p_track.add_argument(
        "--vlm-model",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="HuggingFace VLM id for local escalation (e.g. HuggingFaceTB/SmolVLM-500M-Instruct or HuggingFaceTB/SmolVLM2-2.2B-Instruct).",
    )
    p_track.add_argument("--no-vlm", action="store_true", help="Disable VLM escalation (YOLO+tracker only)")
    p_track.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold")
    p_track.add_argument("--persistence", type=int, default=4, help="Frames needed in window to confirm violation")
    p_track.add_argument("--window", type=int, default=7, help="Sliding window size")
    p_track.add_argument("--skip-frames", type=int, default=1, help="Run YOLO every N frames")
    p_track.add_argument("--save-output", help="Save annotated video to this path")
    p_track.add_argument("--alert-log", help="Save alert log (JSON) to this path")
    p_track.add_argument("--no-display", action="store_true", help="Disable live display window")
    p_track.add_argument("--verbose", action="store_true", help="Print debug info per frame")
    p_track.set_defaults(func=cmd_track)

    # ── check-vlm ─────────────────────────────────────────────────────────────
    p_vlm = sub.add_parser("check-vlm", help="Check if the local VLM backend is available")
    p_vlm.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct", help="Model id to report")
    p_vlm.set_defaults(func=cmd_check_vlm)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
