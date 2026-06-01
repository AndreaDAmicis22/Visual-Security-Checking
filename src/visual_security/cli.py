"""
CLI for Visual Security — PPE Tracker.

Usage examples:

  # Track a video file (YOLO only, no VLM)
  python -m visual_security.cli track \
      --yolo-model weights/dataset_1/yolo_nano_640/best.onnx \
      --source video.mp4

  # Track with VLM escalation (Ollama + moondream)
  python -m visual_security.cli track \
      --yolo-model weights/best.onnx \
      --source video.mp4 \
      --vlm moondream \
      --save-output output/annotated.mp4 \
      --alert-log output/alerts.json

  # Track from webcam
  python -m visual_security.cli track \
      --yolo-model weights/best.onnx \
      --source 0

  # Check if Ollama VLM is available
  python -m visual_security.cli check-vlm --model moondream
"""

from __future__ import annotations

import argparse


def cmd_track(args):
    """Real-time video tracking with YOLO + optional VLM escalation."""
    from .video_tracker import build_tracker

    tracker = build_tracker(
        yolo_model_path=args.yolo_model,
        vlm_model=args.vlm,
        vlm_url=args.vlm_url,
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
    """Verify that Ollama is running and the model is available."""
    from .vlm_validator import OllamaVLMValidator

    vlm = OllamaVLMValidator(model=args.model, base_url=args.vlm_url)

    if vlm.is_available():
        print(f"OK: Ollama model '{args.model}' is available at {args.vlm_url}")
    else:
        print(f"ERROR: model '{args.model}' not found at {args.vlm_url}")
        print("\nSetup:")
        print("  1. Install Ollama: https://ollama.com/download")
        print(f"  2. Pull the model: ollama pull {args.model}")
        print("  3. Ollama starts automatically on http://localhost:11434")


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
        "--vlm", default="none", help="Ollama vision model name (e.g. moondream, minicpm-v, llava-phi3). 'none' to disable."
    )
    p_track.add_argument("--vlm-url", default="http://localhost:11434", help="Ollama server URL")
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
    p_vlm = sub.add_parser("check-vlm", help="Check if Ollama VLM is available")
    p_vlm.add_argument("--model", default="moondream", help="Model name to check")
    p_vlm.add_argument("--vlm-url", default="http://localhost:11434", help="Ollama server URL")
    p_vlm.set_defaults(func=cmd_check_vlm)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
