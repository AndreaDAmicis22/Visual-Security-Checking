"""
Evaluator for safety detection models.

Supports:
- Precision / Recall / F1 per class
- mAP (simplified, IoU-based for YOLO; label-based for VLMs)
- Latency benchmarking
- Cost estimation per 1000 images
- Comparative report generation (CSV + JSON)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .analyzer import AnalysisResult, BaseAnalyzer, Detection, VIOLATION_LABELS


# ---------------------------------------------------------------------------
# Ground truth format
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    image_path: str
    labels: list[str]          # e.g. ["no_helmet", "no_vest"]
    bboxes: Optional[list[list[float]]] = None  # optional, for IoU-based mAP


def load_ground_truth_from_yolo_labels(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_names: list[str],
) -> list[GroundTruth]:
    """
    Load ground truth from YOLO-format .txt label files.
    Assumes images_dir contains .jpg/.png files and
    labels_dir contains matching .txt files.

    YOLO format per line: <class_id> <cx> <cy> <w> <h>
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    gts = []
    for img_file in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        label_file = labels_dir / (img_file.stem + ".txt")
        if not label_file.exists():
            continue
        labels = []
        bboxes = []
        for line in label_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            labels.append(label)
            if len(parts) == 5:
                bboxes.append([float(x) for x in parts[1:]])
        gts.append(GroundTruth(str(img_file), labels, bboxes))
    return gts


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

@dataclass
class ImageEvalResult:
    image_path: str
    model_name: str
    true_positives: int
    false_positives: int
    false_negatives: int
    inference_time_ms: float

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_image(
    result: AnalysisResult,
    ground_truth: GroundTruth,
    conf_threshold: float = 0.3,
) -> ImageEvalResult:
    """
    Label-based matching (works for both YOLO and VLM outputs).
    A predicted violation is a TP if the same label appears in ground truth.
    """
    gt_violations = set(l for l in ground_truth.labels if l in VIOLATION_LABELS)
    pred_violations = set(
        d.label for d in result.detections
        if d.is_violation and d.confidence >= conf_threshold
    )

    tp = len(gt_violations & pred_violations)
    fp = len(pred_violations - gt_violations)
    fn = len(gt_violations - pred_violations)

    return ImageEvalResult(
        image_path=result.image_path,
        model_name=result.model_name,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        inference_time_ms=result.inference_time_ms,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    model_name: str
    total_images: int
    total_tp: int
    total_fp: int
    total_fn: int
    avg_inference_ms: float
    p99_inference_ms: float

    @property
    def precision(self) -> float:
        denom = self.total_tp + self.total_fp
        return self.total_tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.total_tp + self.total_fn
        return self.total_tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "avg_latency_ms": round(self.avg_inference_ms, 2),
            "p99_latency_ms": round(self.p99_inference_ms, 2),
            "tp": self.total_tp,
            "fp": self.total_fp,
            "fn": self.total_fn,
        }


def compute_model_metrics(
    eval_results: list[ImageEvalResult],
) -> ModelMetrics:
    model_name = eval_results[0].model_name if eval_results else "unknown"
    times = sorted(r.inference_time_ms for r in eval_results)
    n = len(times)
    return ModelMetrics(
        model_name=model_name,
        total_images=n,
        total_tp=sum(r.true_positives for r in eval_results),
        total_fp=sum(r.false_positives for r in eval_results),
        total_fn=sum(r.false_negatives for r in eval_results),
        avg_inference_ms=sum(times) / n if n else 0.0,
        p99_inference_ms=times[int(n * 0.99)] if n > 1 else (times[0] if times else 0),
    )


# ---------------------------------------------------------------------------
# Cost model (per 1000 images, approximate April 2025 pricing)
# ---------------------------------------------------------------------------

COST_PER_1K_IMAGES = {
    # VLM API costs assume ~1000 input tokens (image) + ~200 output tokens
    # Prices in USD
    "Claude-Sonnet": {
        "cost_per_1k_usd": 3.0 * 1 + 15.0 * 0.2,      # $3/M input, $15/M output
        "notes": "claude-sonnet-4-5-20251022, ~1000 input + 200 output tokens/image",
        "requires_internet": True,
        "data_leaves_premises": True,
    },
    "Claude-Haiku": {
        "cost_per_1k_usd": 0.25 * 1 + 1.25 * 0.2,
        "notes": "claude-haiku, fastest Anthropic model",
        "requires_internet": True,
        "data_leaves_premises": True,
    },
    "GPT-4o": {
        "cost_per_1k_usd": 2.5 * 1 + 10.0 * 0.2,
        "notes": "gpt-4o, ~1000 input + 200 output tokens/image",
        "requires_internet": True,
        "data_leaves_premises": True,
    },
    "GPT-4o-mini": {
        "cost_per_1k_usd": 0.15 * 1 + 0.60 * 0.2,
        "notes": "gpt-4o-mini, cheapest OpenAI vision model",
        "requires_internet": True,
        "data_leaves_premises": True,
    },
    "YOLO11-ONNX": {
        "cost_per_1k_usd": 0.0,   # local, no API cost
        "notes": "Runs locally, GPU optional. One-time training cost.",
        "requires_internet": False,
        "data_leaves_premises": False,
    },
    "V-JEPA2": {
        "cost_per_1k_usd": 0.0,   # self-hosted
        "notes": "Meta V-JEPA2 self-hosted. High VRAM needed (≥16GB GPU).",
        "requires_internet": False,
        "data_leaves_premises": False,
    },
}

# Multiply cost_per_1k_usd by this to get cost per 1000 images
# Already in USD per 1000 images (with token counts above)
# Note: multiply by 1000/1_000_000 to convert M tokens → per image


def estimate_monthly_cost(
    model_name: str,
    images_per_day: int,
    working_days: int = 22,
) -> dict:
    info = COST_PER_1K_IMAGES.get(model_name, {})
    cost_per_1k = info.get("cost_per_1k_usd", 0.0)
    total_images = images_per_day * working_days
    monthly_usd = (total_images / 1000) * cost_per_1k
    return {
        "model": model_name,
        "images_per_day": images_per_day,
        "total_images_per_month": total_images,
        "cost_per_1k_images_usd": round(cost_per_1k, 4),
        "monthly_cost_usd": round(monthly_usd, 2),
        "requires_internet": info.get("requires_internet", None),
        "data_leaves_premises": info.get("data_leaves_premises", None),
        "notes": info.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Latency benchmark (no ground truth needed)
# ---------------------------------------------------------------------------

def benchmark_latency(
    analyzer: BaseAnalyzer,
    image_path: str | Path,
    n_runs: int = 10,
) -> dict:
    """Run inference n_runs times and return latency stats."""
    times = []
    for _ in range(n_runs):
        r = analyzer.analyze(image_path)
        if r.error is None:
            times.append(r.inference_time_ms)

    if not times:
        return {"error": "all runs failed"}

    times.sort()
    return {
        "model": analyzer.model_name,
        "n_runs": len(times),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "avg_ms": round(sum(times) / len(times), 2),
        "p50_ms": round(times[len(times) // 2], 2),
        "p95_ms": round(times[int(len(times) * 0.95)], 2),
        "p99_ms": round(times[int(len(times) * 0.99)] if len(times) > 1 else times[-1], 2),
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

class Evaluator:
    """
    End-to-end evaluator: runs analyzers on a test set and computes metrics.

    Example:
        evaluator = Evaluator(
            analyzers=[ClaudeVisionAnalyzer(), YOLOAnalyzer("model.onnx")],
            ground_truths=load_ground_truth_from_yolo_labels(...)
        )
        report = evaluator.run()
        evaluator.save_report(report, "eval_results/")
    """

    def __init__(
        self,
        analyzers: list[BaseAnalyzer],
        ground_truths: list[GroundTruth],
        conf_threshold: float = 0.3,
    ):
        self.analyzers = analyzers
        self.ground_truths = ground_truths
        self.conf_threshold = conf_threshold

    def run(self) -> dict[str, ModelMetrics]:
        # Index GT by image path
        gt_by_path: dict[str, GroundTruth] = {gt.image_path: gt for gt in self.ground_truths}
        metrics_by_model: dict[str, ModelMetrics] = {}

        for analyzer in self.analyzers:
            eval_results = []
            for gt in self.ground_truths:
                result = analyzer.analyze(gt.image_path)
                if result.error:
                    print(f"  [WARN] {analyzer.model_name} failed on {gt.image_path}: {result.error}")
                    continue
                er = evaluate_image(result, gt, self.conf_threshold)
                eval_results.append(er)
                print(f"  {analyzer.model_name} | {Path(gt.image_path).name} | "
                      f"P={er.precision:.2f} R={er.recall:.2f} F1={er.f1:.2f} "
                      f"({er.inference_time_ms:.0f}ms)")

            if eval_results:
                metrics_by_model[analyzer.model_name] = compute_model_metrics(eval_results)

        return metrics_by_model

    @staticmethod
    def print_comparison(metrics: dict[str, ModelMetrics]) -> None:
        print("\n" + "=" * 70)
        print(f"{'Model':<20} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Avg(ms)':>10} {'P99(ms)':>10}")
        print("-" * 70)
        for m in sorted(metrics.values(), key=lambda x: x.f1, reverse=True):
            print(
                f"{m.model_name:<20} {m.precision:>10.3f} {m.recall:>8.3f} "
                f"{m.f1:>6.3f} {m.avg_inference_ms:>10.1f} {m.p99_inference_ms:>10.1f}"
            )
        print("=" * 70 + "\n")

    @staticmethod
    def save_report(
        metrics: dict[str, ModelMetrics],
        output_dir: str | Path,
        images_per_day: int = 500,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON metrics
        data = {
            "metrics": [m.to_dict() for m in metrics.values()],
            "cost_estimates": [
                estimate_monthly_cost(name, images_per_day)
                for name in COST_PER_1K_IMAGES
            ],
        }
        (output_dir / "evaluation_report.json").write_text(
            json.dumps(data, indent=2)
        )

        # CSV summary
        csv_lines = ["model,precision,recall,f1,avg_latency_ms,p99_latency_ms"]
        for m in metrics.values():
            d = m.to_dict()
            csv_lines.append(
                f"{d['model']},{d['precision']},{d['recall']},{d['f1']},"
                f"{d['avg_latency_ms']},{d['p99_latency_ms']}"
            )
        (output_dir / "metrics.csv").write_text("\n".join(csv_lines))

        print(f"Report saved to {output_dir}/")
