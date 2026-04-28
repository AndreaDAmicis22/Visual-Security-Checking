"""
CLI for Visual Security Checking.

Usage examples:

  # Analyze a single image with Claude Vision
  python -m src.visual_security.cli analyze --image site.jpg --model claude

  # Analyze with all available models
  python -m src.visual_security.cli analyze --image site.jpg --model all

  # Run evaluation on a labeled test set
  python -m src.visual_security.cli evaluate \
      --images data/test/images \
      --labels data/test/labels \
      --model all

  # Benchmark latency
  python -m src.visual_security.cli benchmark --image site.jpg --model yolo --runs 20

  # Show cost estimates
  python -m src.visual_security.cli costs --images-per-day 500
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np


def _build_analyzers(model_arg: str, yolo_model: str | None = None):
    from src.visual_security.analyzer import (
        ClaudeVisionAnalyzer,
        GPT4oAnalyzer,
        YOLOAnalyzer,
    )

    analyzers = []
    if model_arg in ("claude", "all"):
        analyzers.append(ClaudeVisionAnalyzer())
    if model_arg in ("gpt4o", "all"):
        analyzers.append(GPT4oAnalyzer())
    if model_arg in ("yolo", "all") and yolo_model:
        analyzers.append(YOLOAnalyzer(yolo_model))
    elif model_arg == "yolo" and not yolo_model:
        print("[WARN] --yolo-model not provided, skipping YOLO")
    return analyzers


def cmd_analyze(args):
    analyzers = _build_analyzers(args.model, args.yolo_model)
    if not analyzers:
        print("No analyzers configured. Check API keys or --yolo-model path.")
        return

    from src.visual_security.analyzer import SafetyAnalyzerPipeline
    pipeline = SafetyAnalyzerPipeline(analyzers)
    results = pipeline.run(args.image)
    pipeline.print_report(results)

    if args.visualize:
        _draw_results(args.image, results)


def cmd_evaluate(args):
    from src.visual_security.analyzer import YOLOAnalyzer, ClaudeVisionAnalyzer, GPT4oAnalyzer
    from src.visual_security.evaluator import (
        Evaluator,
        load_ground_truth_from_yolo_labels,
    )

    # Class names: try to load from data.yaml, fall back to default
    class_names = _load_class_names(args.data_yaml)

    gt = load_ground_truth_from_yolo_labels(args.images, args.labels, class_names)
    print(f"Loaded {len(gt)} ground truth samples.")

    analyzers = _build_analyzers(args.model, args.yolo_model)
    evaluator = Evaluator(analyzers, gt, conf_threshold=args.conf)
    metrics = evaluator.run()

    Evaluator.print_comparison(metrics)
    if args.output:
        Evaluator.save_report(metrics, args.output, images_per_day=args.images_per_day)


def cmd_benchmark(args):
    analyzers = _build_analyzers(args.model, args.yolo_model)
    from src.visual_security.evaluator import benchmark_latency
    for analyzer in analyzers:
        result = benchmark_latency(analyzer, args.image, n_runs=args.runs)
        print(json.dumps(result, indent=2))


def cmd_costs(args):
    from src.visual_security.evaluator import COST_PER_1K_IMAGES, estimate_monthly_cost
    print(f"\nMonthly cost estimate for {args.images_per_day} images/day ({args.working_days} working days):\n")
    print(f"{'Model':<20} {'$/1k images':>12} {'Monthly $':>12} {'Internet':>10} {'Data off-prem':>14}")
    print("-" * 72)
    for model_name in COST_PER_1K_IMAGES:
        est = estimate_monthly_cost(model_name, args.images_per_day, args.working_days)
        internet = "Yes" if est["requires_internet"] else "No"
        data_off = "Yes" if est["data_leaves_premises"] else "No"
        print(
            f"{model_name:<20} {est['cost_per_1k_images_usd']:>12.4f} "
            f"{est['monthly_cost_usd']:>12.2f} {internet:>10} {data_off:>14}"
        )
    print()


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def _draw_results(image_path: str, results) -> None:
    """Draw bounding boxes and labels on the image and show it."""
    img = cv.imread(image_path)
    if img is None:
        return

    colors = {
        True: (0, 0, 255),   # Red for violations
        False: (0, 255, 0),  # Green for safe
    }

    y_offset = 30
    for result in results:
        label = f"{result.model_name}: {len(result.violations)} violation(s)"
        cv.putText(img, label, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 0), 2)
        y_offset += 25

        for det in result.detections:
            if det.bbox and len(det.bbox) >= 4:
                pts = np.array(det.bbox, dtype=np.int32)
                color = colors[det.is_violation]
                cv.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
                text = f"{det.label} {det.confidence:.2f}"
                x, y = int(pts[0][0]), int(pts[0][1]) - 5
                cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 1)

    cv.imshow("Safety Analysis", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def _load_class_names(data_yaml_path: str | None) -> list[str]:
    if not data_yaml_path:
        from src.visual_security.analyzer import YOLOAnalyzer
        return YOLOAnalyzer.DEFAULT_CLASS_NAMES

    try:
        import yaml
        with open(data_yaml_path) as f:
            data = yaml.safe_load(f)
        return data.get("names", [])
    except Exception:
        from src.visual_security.analyzer import YOLOAnalyzer
        return YOLOAnalyzer.DEFAULT_CLASS_NAMES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="visual-security",
        description="Construction site safety analyzer — multi-model evaluation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Analyze a single image")
    p_analyze.add_argument("--image", required=True, help="Path to image file")
    p_analyze.add_argument(
        "--model", default="claude",
        choices=["claude", "gpt4o", "yolo", "all"],
        help="Which model(s) to use",
    )
    p_analyze.add_argument("--yolo-model", default=None, help="Path to YOLO ONNX model file")
    p_analyze.add_argument("--visualize", action="store_true", help="Show annotated image")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Evaluate on labeled test set")
    p_eval.add_argument("--images", required=True, help="Directory with test images")
    p_eval.add_argument("--labels", required=True, help="Directory with YOLO label .txt files")
    p_eval.add_argument("--data-yaml", default=None, help="Path to data.yaml (for class names)")
    p_eval.add_argument("--model", default="all", choices=["claude", "gpt4o", "yolo", "all"])
    p_eval.add_argument("--yolo-model", default=None)
    p_eval.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    p_eval.add_argument("--output", default="eval_results", help="Output directory")
    p_eval.add_argument("--images-per-day", type=int, default=500)

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Latency benchmark")
    p_bench.add_argument("--image", required=True)
    p_bench.add_argument("--model", default="claude", choices=["claude", "gpt4o", "yolo", "all"])
    p_bench.add_argument("--yolo-model", default=None)
    p_bench.add_argument("--runs", type=int, default=10)

    # --- costs ---
    p_costs = sub.add_parser("costs", help="Show cost estimates")
    p_costs.add_argument("--images-per-day", type=int, default=500)
    p_costs.add_argument("--working-days", type=int, default=22)

    args = parser.parse_args()
    {
        "analyze": cmd_analyze,
        "evaluate": cmd_evaluate,
        "benchmark": cmd_benchmark,
        "costs": cmd_costs,
    }[args.command](args)


if __name__ == "__main__":
    main()
