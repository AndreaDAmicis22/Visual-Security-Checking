"""
CLI for Visual Security — PPE Tracker.

Usage examples:

  # Track a video file (detector only, no VLM)
  python -m visual_security.cli track \
      --source video.mp4 \
      --no-vlm

  # Track with restricted zones + local VLM escalation
  python -m visual_security.cli track \
      --source video.mp4 \
      --zones zones.json \
      --save-output output/annotated.mp4 \
      --alert-log output/alerts.json

  # Faster detector (OmDet-Turbo) from webcam
  python -m visual_security.cli track \
      --source 0 \
      --detector omdet-turbo

  # Check if the inference backend (torch/transformers) is available
  python -m visual_security.cli check-vlm
"""

from __future__ import annotations

import argparse


def cmd_track(args):
    """Real-time video tracking: open-vocab detector + zones + optional VLM escalation."""
    from .video_tracker import build_tracker

    tracker = build_tracker(
        detector=args.detector,
        vlm_model="none" if args.no_vlm else args.vlm_model,
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


def cmd_check_vlm(args):
    """Verify that the local inference backend (torch/transformers) is importable."""
    from .vlm_validator import LocalVLMValidator

    if LocalVLMValidator.is_available():
        print("OK: torch + transformers disponibili. Detector e VLM locali utilizzabili.")
        print(f"    VLM di default: '{args.model}' (scaricato al primo utilizzo).")
    else:
        print("ERROR: torch/transformers non installati.")
        print("\nSetup:")
        print("  pip install torch transformers pillow")


def main():
    parser = argparse.ArgumentParser(
        prog="visual_security",
        description="PPE Safety Tracker - detector open-vocabulary (Apache 2.0) + zone vietate + VLM escalation",
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
    p_track.add_argument(
        "--vlm-model",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="HuggingFace VLM id for local escalation (e.g. HuggingFaceTB/SmolVLM-500M-Instruct or HuggingFaceTB/SmolVLM2-2.2B-Instruct).",
    )
    p_track.add_argument("--no-vlm", action="store_true", help="Disable VLM escalation (detector+tracker only)")
    p_track.add_argument("--conf", type=float, default=None, help="Detector confidence threshold (default: backend default)")
    p_track.add_argument("--persistence", type=int, default=4, help="Frames needed in window to confirm violation")
    p_track.add_argument("--window", type=int, default=7, help="Sliding window size")
    p_track.add_argument("--skip-frames", type=int, default=1, help="Run the detector every N frames")
    p_track.add_argument("--save-output", help="Save annotated video to this path")
    p_track.add_argument("--alert-log", help="Save alert log (JSON) to this path")
    p_track.add_argument("--no-display", action="store_true", help="Disable live display window")
    p_track.add_argument("--verbose", action="store_true", help="Print debug info per frame")
    p_track.set_defaults(func=cmd_track)

    # ── check-vlm ─────────────────────────────────────────────────────────────
    p_vlm = sub.add_parser("check-vlm", help="Check if the local inference backend is available")
    p_vlm.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct", help="Model id to report")
    p_vlm.set_defaults(func=cmd_check_vlm)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
