from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_acl.data import ECGNpzDataset, SyntheticECGDataset
from ecg_acl.metrics import ReliabilityZoneMeter
from ecg_acl.model import build_ecg_model, unpack_model_output
from ecg_acl.train_utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ECG classifier checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="ecg_acl/data/mitbih/processed/test.npz")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--synthetic", action="store_true", help="Evaluate on synthetic smoke-test data.")
    parser.add_argument(
        "--decision-rule",
        choices=["argmax", "threshold", "reject"],
        default="argmax",
        help="Post-processing rule applied to class probabilities.",
    )
    parser.add_argument(
        "--tune-data",
        default="ecg_acl/data/mitbih/processed/val.npz",
        help="Validation npz used to tune class thresholds.",
    )
    parser.add_argument(
        "--target-metric",
        choices=["acc", "macro_f1", "balanced_acc", "minority_recall", "minority_f1"],
        default="balanced_acc",
        help="Metric optimized when tuning thresholds.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated class thresholds or a JSON file with a `thresholds` array.",
    )
    parser.add_argument("--reject-threshold", type=float, default=None, help="Reject samples below this max probability.")
    parser.add_argument("--tune-thresholds", action="store_true", help="Tune class thresholds on --tune-data.")
    parser.add_argument("--save-decision-config", default=None, help="Optional JSON output for tuned thresholds.")
    return parser.parse_args()


def _load_thresholds(value: str | None, num_classes: int) -> np.ndarray | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        thresholds = data.get("thresholds", data)
    else:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    arr = np.asarray(thresholds, dtype=np.float64)
    if arr.shape != (num_classes,):
        raise ValueError(f"Expected {num_classes} thresholds, got {arr.tolist()}")
    return np.clip(arr, 1e-6, 1.0)


def _dataset_from_args(path: str, synthetic: bool, cfg: dict, seed_offset: int = 999):
    if synthetic:
        return SyntheticECGDataset(
            size=64,
            num_classes=int(cfg["data"]["num_classes"]),
            channels=int(cfg["data"]["input_channels"]),
            length=int(cfg["data"]["input_length"]),
            seed=int(cfg["seed"]) + seed_offset,
        )
    return ECGNpzDataset(path)


@torch.no_grad()
def collect_probabilities(model, dataset, batch_size: int, device: torch.device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = []
    targets = []
    reliability = ReliabilityZoneMeter()
    for x, y in loader:
        logits, _, aux_logits = unpack_model_output(model(x.to(device)))
        reliability.update(aux_logits, y.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets), reliability.compute()


def predict_argmax(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1)


def predict_with_thresholds(
    probs: np.ndarray,
    thresholds: np.ndarray,
    reject_threshold: float | None = None,
    reject_label: int | None = None,
) -> np.ndarray:
    ratios = probs / thresholds.reshape(1, -1)
    preds = ratios.argmax(axis=1)
    if reject_threshold is not None:
        if reject_label is None:
            raise ValueError("reject_label is required when reject_threshold is set")
        preds = preds.copy()
        preds[probs.max(axis=1) < reject_threshold] = reject_label
    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    reject_label = num_classes
    has_reject = bool((y_pred == reject_label).any())
    pred_cols = num_classes + 1 if has_reject else num_classes
    matrix = np.zeros((num_classes, pred_cols), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        if pred < 0 or pred >= pred_cols:
            raise ValueError(f"Prediction out of range: {pred}")
        matrix[int(target), int(pred)] += 1

    tp = np.diag(matrix[:, :num_classes]).astype(np.float64)
    support = matrix.sum(axis=1).clip(min=1).astype(np.float64)
    pred_count = matrix[:, :num_classes].sum(axis=0).clip(min=1).astype(np.float64)
    recall = tp / support
    precision = tp / pred_count
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    covered = y_pred != reject_label
    covered_total = max(int(covered.sum()), 1)
    metrics = {
        "confusion_matrix": matrix.tolist(),
        "accuracy": float((y_true == y_pred).mean()),
        "acc": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(recall.mean()),
        "balanced_acc": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "minority_recall": float(recall[1:].mean()) if num_classes > 1 else float(recall.mean()),
        "minority_f1": float(f1[1:].mean()) if num_classes > 1 else float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "coverage": float(covered.mean()),
        "reject_rate": float(1.0 - covered.mean()),
        "covered_accuracy": float((y_true[covered] == y_pred[covered]).mean()) if covered.any() else 0.0,
        "covered_samples": int(covered_total),
        "samples": int(len(y_true)),
    }
    return metrics


def _metric_value(metrics: dict, metric_name: str) -> float:
    return float(metrics[metric_name])


def tune_thresholds(probs: np.ndarray, y_true: np.ndarray, num_classes: int, target_metric: str) -> tuple[np.ndarray, float]:
    thresholds = np.full(num_classes, 0.5, dtype=np.float64)
    grid = np.asarray([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.80])
    best_metrics = compute_metrics(y_true, predict_with_thresholds(probs, thresholds), num_classes)
    best_score = _metric_value(best_metrics, target_metric)
    for _ in range(3):
        improved = False
        for cls in range(num_classes):
            class_best = thresholds[cls]
            for value in grid:
                candidate = thresholds.copy()
                candidate[cls] = value
                pred = predict_with_thresholds(probs, candidate)
                metrics = compute_metrics(y_true, pred, num_classes)
                score = _metric_value(metrics, target_metric)
                if score > best_score:
                    best_score = score
                    class_best = value
                    improved = True
            thresholds[cls] = class_best
        if not improved:
            break
    return thresholds, best_score


def print_metrics(metrics: dict, reliability_zones: dict | None = None) -> None:
    print("confusion_matrix:")
    print(np.asarray(metrics["confusion_matrix"]))
    print("per_class_recall:", np.asarray(metrics["per_class_recall"]))
    print("per_class_f1:", np.asarray(metrics["per_class_f1"]))
    print("accuracy:", metrics["accuracy"])
    print("balanced_accuracy:", metrics["balanced_accuracy"])
    print("macro_f1:", metrics["macro_f1"])
    print("minority_recall:", metrics["minority_recall"])
    print("minority_f1:", metrics["minority_f1"])
    print("coverage:", metrics["coverage"])
    print("reject_rate:", metrics["reject_rate"])
    print("covered_accuracy:", metrics["covered_accuracy"])
    if reliability_zones:
        print("reliability_zones:", reliability_zones)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_ecg_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    num_classes = int(cfg["data"]["num_classes"])
    thresholds = _load_thresholds(args.thresholds, num_classes)
    if args.tune_thresholds:
        tune_dataset = _dataset_from_args(args.tune_data, args.synthetic, cfg, seed_offset=777)
        tune_probs, tune_y, _ = collect_probabilities(model, tune_dataset, args.batch_size, device)
        thresholds, tune_score = tune_thresholds(tune_probs, tune_y, num_classes, args.target_metric)
        print(f"tuned_thresholds={thresholds.tolist()} target_metric={args.target_metric} val_score={tune_score:.6f}")
        if args.save_decision_config:
            Path(args.save_decision_config).write_text(
                json.dumps(
                    {
                        "decision_rule": args.decision_rule,
                        "target_metric": args.target_metric,
                        "thresholds": thresholds.tolist(),
                        "reject_threshold": args.reject_threshold,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    dataset = _dataset_from_args(args.data, args.synthetic, cfg)
    probs, y_true, zones = collect_probabilities(model, dataset, args.batch_size, device)
    if args.decision_rule == "argmax":
        y_pred = predict_argmax(probs)
    elif args.decision_rule == "threshold":
        if thresholds is None:
            thresholds = np.full(num_classes, 0.5, dtype=np.float64)
        y_pred = predict_with_thresholds(probs, thresholds)
    else:
        if thresholds is None:
            thresholds = np.ones(num_classes, dtype=np.float64)
        reject_threshold = 0.5 if args.reject_threshold is None else float(args.reject_threshold)
        y_pred = predict_with_thresholds(probs, thresholds, reject_threshold, reject_label=num_classes)

    metrics = compute_metrics(y_true, y_pred, num_classes)
    print_metrics(metrics, zones)


if __name__ == "__main__":
    main()
