from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CLASS_NAMES = ["N", "S", "V", "F", "Q"]
SCALAR_METRICS = ["acc", "balanced_acc", "macro_f1", "minority_recall", "minority_f1"]
PER_CLASS_METRICS = ["per_class_precision", "per_class_recall", "per_class_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ECG experiment metrics from metrics.json.")
    parser.add_argument(
        "--metrics",
        default="ecg_acl/work_dirs/fusion_context_rr_record_acl/metrics.json",
        help="Path to a metrics.json file or a work directory containing metrics.json.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to <work_dir>/visualizations.",
    )
    parser.add_argument(
        "--class-names",
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names for per-class plots.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title prefix for generated figures.",
    )
    return parser.parse_args()


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to generate visualizations: pip install matplotlib") from exc
    return plt


def _metrics_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"metrics.json not found: {path}")
    return path


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "history" not in data or "test" not in data:
        raise ValueError(f"Expected metrics.json with `history` and `test`: {path}")
    return data


def _safe_metric(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _plot_training_curves(data: dict[str, Any], out_dir: Path, title: str | None) -> Path:
    plt = _import_matplotlib()
    history = data["history"]
    epochs = [int(item["epoch"]) for item in history]
    panels = [
        ("Loss", ["loss", "ce", "aux"]),
        ("Accuracy", ["acc", "balanced_acc"]),
        ("F1", ["macro_f1", "minority_f1"]),
        ("Minority Recall", ["minority_recall"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=140)
    fig.suptitle(title or "Training Curves")
    for ax, (panel_title, keys) in zip(axes.ravel(), panels):
        for split in ["train", "val"]:
            for key in keys:
                values = [_safe_metric(item[split], key) for item in history]
                if any(value is not None for value in values):
                    ax.plot(epochs, values, label=f"{split}_{key}", linewidth=1.8)
        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_final_scalar_metrics(data: dict[str, Any], out_dir: Path, title: str | None) -> Path:
    plt = _import_matplotlib()
    test = data["test"]
    labels = []
    values = []
    for key in SCALAR_METRICS:
        value = _safe_metric(test, key)
        if value is not None:
            labels.append(key)
            values.append(value)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.bar(labels, values, color=["#4062bb", "#59a14f", "#f28e2b", "#e15759", "#76b7b2"][: len(values)])
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title or "Final Test Metrics")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = out_dir / "final_scalar_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_per_class_metrics(data: dict[str, Any], out_dir: Path, class_names: list[str], title: str | None) -> Path:
    plt = _import_matplotlib()
    test = data["test"]
    values = []
    labels = []
    for key in PER_CLASS_METRICS:
        metric_values = test.get(key)
        if isinstance(metric_values, list):
            labels.append(key.replace("per_class_", ""))
            values.append([float(item) for item in metric_values])
    if not values:
        raise ValueError("No per-class metrics found in test results.")

    arr = np.asarray(values, dtype=np.float64)
    x = np.arange(arr.shape[1])
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    for idx, label in enumerate(labels):
        ax.bar(x + (idx - (len(labels) - 1) / 2) * width, arr[idx], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names[: arr.shape[1]])
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title or "Per-Class Test Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "per_class_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusion_matrix(data: dict[str, Any], out_dir: Path, class_names: list[str], title: str | None) -> Path:
    plt = _import_matplotlib()
    matrix = np.asarray(data["test"].get("confusion"), dtype=np.float64)
    if matrix.ndim != 2:
        matrix = np.asarray(data["test"].get("confusion_matrix"), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("No confusion matrix found in test results.")

    row_sum = np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
    normalized = matrix / row_sum
    x_labels = class_names[: matrix.shape[1]]
    if matrix.shape[1] > len(x_labels):
        x_labels += [f"pred_{idx}" for idx in range(len(x_labels), matrix.shape[1])]
    y_labels = class_names[: matrix.shape[0]]

    fig, ax = plt.subplots(figsize=(7.2, 6), dpi=140)
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=max(0.01, normalized.max()))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title or "Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticklabels(y_labels)
    if matrix.size <= 64:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(
                    col,
                    row,
                    f"{int(matrix[row, col])}\n{normalized[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalized[row, col] > normalized.max() * 0.55 else "black",
                )
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_topk_csv(data: dict[str, Any], out_dir: Path) -> Path:
    candidates = data.get("checkpoint_candidates", {})
    rows = []
    if isinstance(candidates, dict):
        for metric, items in candidates.items():
            for item in items:
                rows.append(
                    {
                        "metric": metric,
                        "epoch": item.get("epoch"),
                        "score": item.get("score"),
                        "path": item.get("path"),
                    }
                )
    elif isinstance(candidates, list):
        for item in candidates:
            rows.append(
                {
                    "metric": item.get("metric"),
                    "epoch": item.get("epoch"),
                    "score": item.get("score"),
                    "path": item.get("path"),
                }
            )

    path = out_dir / "topk_checkpoints.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "epoch", "score", "path"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_summary(data: dict[str, Any], out_dir: Path, generated: list[Path]) -> Path:
    test = data["test"]
    best_checkpoints = data.get("best_checkpoints", {})
    lines = [
        "# Experiment Summary",
        "",
        f"- Best epoch: {data.get('best_epoch')}",
        f"- Primary selection metric: {data.get('selection_metric')}",
        f"- Best score: {data.get('best_score')}",
        "",
        "## Test Metrics",
        "",
    ]
    for key in SCALAR_METRICS:
        value = _safe_metric(test, key)
        if value is not None:
            lines.append(f"- {key}: {value:.6f}")
    if best_checkpoints:
        lines += ["", "## Best Checkpoints", ""]
        for metric, item in best_checkpoints.items():
            lines.append(f"- {metric}: epoch {item.get('epoch')} score {float(item.get('score', 0.0)):.6f}")
    lines += ["", "## Generated Files", ""]
    for path in generated:
        lines.append(f"- {path.name}")

    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    metrics_path = _metrics_path(args.metrics)
    out_dir = Path(args.out) if args.out else metrics_path.parent / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    data = _load_metrics(metrics_path)

    title_prefix = args.title or metrics_path.parent.name
    generated = [
        _plot_training_curves(data, out_dir, f"{title_prefix} Training Curves"),
        _plot_final_scalar_metrics(data, out_dir, f"{title_prefix} Final Test Metrics"),
        _plot_per_class_metrics(data, out_dir, class_names, f"{title_prefix} Per-Class Metrics"),
        _plot_confusion_matrix(data, out_dir, class_names, f"{title_prefix} Confusion Matrix"),
        _write_topk_csv(data, out_dir),
    ]
    generated.append(_write_summary(data, out_dir, generated))
    print(f"visualizations written to: {out_dir}")


if __name__ == "__main__":
    main()
