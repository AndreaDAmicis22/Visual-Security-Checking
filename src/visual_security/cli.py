"""
CLI for Visual Security — PPE Tracker.

Usage examples:

  # Track a video file
  python -m visual_security.cli track --source video.mp4

  # Track with restricted zones + annotated output
  python -m visual_security.cli track \
      --source video.mp4 \
      --zones zones.example.json \
      --save-output output/annotated.mp4 \
      --alert-log output/alerts.json

  # Faster detector (OmDet-Turbo) from webcam
  python -m visual_security.cli track \
      --source 0 \
      --detector omdet-turbo

  # Check if the inference backend (torch/transformers) is available
  python -m visual_security.cli check-backend
"""

from __future__ import annotations

import argparse


def cmd_track(args):
    """Real-time video tracking: open-vocab detector + zones."""
    from .video_tracker import build_tracker

    tracker = build_tracker(
        detector=args.detector,
        zones_file=args.zones,
        persistence_frames=args.persistence,
        window_frames=args.window,
        skip_frames=args.skip_frames,
        display=not args.no_display,
        save_output=args.save_output,
        alert_log=args.alert_log,
        detector_conf=args.conf,
        verbose=args.verbose,
    )

    source = int(args.source) if args.source.isdigit() else args.source
    alerts = tracker.run(source)

    print(f"\n{'=' * 60}")
    print(f"Tracking complete - {len(alerts)} confirmed alert(s).")
    print(f"{'=' * 60}")
    for a in alerts:
        print(a.summary())


def cmd_check_backend(args):  # noqa: ARG001
    """Verify that the local inference backend (torch/transformers) is importable."""
    try:
        import torch
        import transformers

        print(f"OK: torch {torch.__version__} + transformers {transformers.__version__} disponibili.")
        print("    I pesi dei detector vengono scaricati da HuggingFace al primo utilizzo.")
    except ImportError as e:
        print(f"ERROR: backend non disponibile ({e}).")
        print("\nSetup:")
        print("  pip install torch transformers pillow timm")


def main():
    parser = argparse.ArgumentParser(
        prog="visual_security",
        description="PPE Safety Tracker - detector open-vocabulary (Apache 2.0) + zone vietate",
    )
    sub = parser.add_subparsers(dest="command")

    # ── track ─────────────────────────────────────────────────────────────────
    p_track = sub.add_parser("track", help="Run real-time PPE + zone tracking on video/webcam")
    p_track.add_argument("--source", default="0", help="Video file path or camera index (default: 0)")
    p_track.add_argument(
        "--detector",
        default="grounding-dino",
        choices=["grounding-dino", "omdet-turbo"],
        help="Open-vocabulary detector: grounding-dino (max accuracy) or omdet-turbo (faster).",
    )
    p_track.add_argument("--zones", help="JSON file with restricted zones (see zone_monitor.py for format)")
    p_track.add_argument("--conf", type=float, default=None, help="Detector confidence threshold (default: backend default)")
    p_track.add_argument("--persistence", type=int, default=4, help="Frames needed in window to confirm violation")
    p_track.add_argument("--window", type=int, default=7, help="Sliding window size")
    p_track.add_argument("--skip-frames", type=int, default=1, help="Run the detector every N frames")
    p_track.add_argument("--save-output", help="Save annotated video to this path")
    p_track.add_argument("--alert-log", help="Save alert log (JSON) to this path")
    p_track.add_argument("--no-display", action="store_true", help="Disable live display window")
    p_track.add_argument("--verbose", action="store_true", help="Print debug info per frame")
    p_track.set_defaults(func=cmd_track)

    # ── check-backend ─────────────────────────────────────────────────────────
    p_chk = sub.add_parser("check-backend", help="Check if the local inference backend is available")
    p_chk.set_defaults(func=cmd_check_backend)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
