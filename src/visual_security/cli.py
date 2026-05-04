"""
CLI for Visual Security Checking.

Usage examples:

  # Analyze a single image with Foundry GPT-4o
  python -m src.visual_security.cli analyze --image site.jpg --model gpt4o

  # Analyze with all available models (reads keys from .env)
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

Environment variables (put in .env file):
  AZURE_OPENAI_KEY      — API key for Foundry GPT-4o
  AZURE_OPENAI_URL      — Full endpoint URL for Foundry GPT-4o deployment
  AZURE_VISION_KEY      — API key for Azure AI Vision
  AZURE_VISION_ENDPOINT — Base endpoint for Azure AI Vision
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def _build_analyzers(model_arg: str, yolo_model: str | None = None):
    from src.visual_security.analyzer import (
        AzureAIVisionAnalyzer,
        FoundryGPT4oAnalyzer,
        YOLOAnalyzer,
    )

    analyzers = []

    if model_arg in ("yolo", "all"):
        if yolo_model and Path(yolo_model).exists():
            analyzers.append(YOLOAnalyzer(model_path=yolo_model))
        else:
            print("[WARN] --yolo-model not found or not provided, skipping YOLO")

    if model_arg in ("gpt4o", "all"):
        key = os.getenv("AZURE_OPENAI_KEY")
        url = os.getenv("AZURE_OPENAI_URL")
        if key:
            analyzers.append(FoundryGPT4oAnalyzer(api_key=key, endpoint=url))
        else:
            print("[WARN] AZURE_OPENAI_KEY not set, skipping Foundry GPT-4o")

    if model_arg in ("vision", "all"):
        key = os.getenv("AZURE_VISION_KEY")
        endpoint = os.getenv("AZURE_VISION_ENDPOINT")
        if key:
            analyzers.append(AzureAIVisionAnalyzer(api_key=key, endpoint=endpoint))
        else:
            print("[WARN] AZURE_VISION_KEY not set, skipping Azure AI Vision")

    return analyzers


def cmd_analyze(args):
    analyzers = _build_analyzers(args.model, args.yolo_model)
    if not analyzers:
        print("No analyzers configured. Check your .env file or --yolo-model path.")
        return

    from src.visual_security.analyzer import SafetyAnalyzerPipeline

    pipeline = SafetyAnalyzerPipeline(analyzers)
    results = pipeline.run(args.image)
    pipeline.print_report(results)

    if args.visualize:
        _draw_results(args.image, results)


def cmd_evaluate(args):
    from src.visual_security.evaluator import (
        Evaluator,
        load_ground_truth_from_yolo_labels,
    )

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

    print(f"\nMonthly cost estimate — {args.images_per_day} images/day, {args.working_days} working days:\n")
    print(f"{'Model':<25} {'$/1k images':>12} {'Monthly $':>12} {'Internet':>10} {'Data off-prem':>14}")
    print("-" * 77)
    for model_name in COST_PER_1K_IMAGES:
        est = estimate_monthly_cost(model_name, args.images_per_day, args.working_days)
        internet = "Yes" if est["requires_internet"] else "No"
        data_off = "Yes" if est["data_leaves_premises"] else "No"
        print(
            f"{model_name:<25} {est['cost_per_1k_images_usd']:>12.4f} "
            f"{est['monthly_cost_usd']:>12.2f} {internet:>10} {data_off:>14}"
        )
    print()


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------
def _draw_results(image_path: str, results) -> None:
    img = cv.imread(image_path)
    if img is None:
        return

    RED = (0, 0, 220)
    GREEN = (30, 200, 80)
    YELLOW = (20, 200, 220)

    y = 30
    for result in results:
        cv.putText(
            img,
            f"{result.model_name}: {len(result.violations)} violation(s)",
            (10, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            YELLOW,
            1,
            cv.LINE_AA,
        )
        y += 24
        for det in result.detections:
            color = RED if det.is_violation else GREEN
            if det.bbox and len(det.bbox) >= 3:
                pts = np.array(det.bbox, dtype=np.int32)
                cv.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
                lx, ly = int(pts[0][0]), max(int(pts[0][1]) - 6, 12)
                cv.putText(
                    img, f"{det.label} {det.confidence:.2f}", (lx, ly), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA
                )

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
            return yaml.safe_load(f).get("names", [])
    except Exception:
        from src.visual_security.analyzer import YOLOAnalyzer

        return YOLOAnalyzer.DEFAULT_CLASS_NAMES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="visual-security",
        description="Construction site safety analyzer — Foundry + YOLO",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p = sub.add_parser("analyze", help="Analyze a single image")
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="gpt4o", choices=["yolo", "gpt4o", "vision", "all"])
    p.add_argument("--yolo-model", default=None)
    p.add_argument("--visualize", action="store_true")

    # evaluate
    p = sub.add_parser("evaluate", help="Evaluate on labeled test set")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--model", default="all", choices=["yolo", "gpt4o", "vision", "all"])
    p.add_argument("--yolo-model", default=None)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--output", default="eval_results")
    p.add_argument("--images-per-day", type=int, default=500)

    # benchmark
    p = sub.add_parser("benchmark", help="Latency benchmark")
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="gpt4o", choices=["yolo", "gpt4o", "vision", "all"])
    p.add_argument("--yolo-model", default=None)
    p.add_argument("--runs", type=int, default=10)

    # costs
    p = sub.add_parser("costs", help="Show cost estimates")
    p.add_argument("--images-per-day", type=int, default=500)
    p.add_argument("--working-days", type=int, default=22)

    args = parser.parse_args()
    {"analyze": cmd_analyze, "evaluate": cmd_evaluate, "benchmark": cmd_benchmark, "costs": cmd_costs}[args.command](args)


if __name__ == "__main__":
    main()
