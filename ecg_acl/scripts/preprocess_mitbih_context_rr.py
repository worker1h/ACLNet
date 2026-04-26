from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample

from preprocess_mitbih import (
    AAMI_CLASSES,
    AAMI_MAP,
    CLASS_TO_ID,
    DS1_RECORDS,
    DS2_RECORDS,
    _normalize_beat,
    _record_ids,
    _stratified_val_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MIT-BIH with neighbor beats and RR interval channels.")
    parser.add_argument("--raw", default="ecg_acl/data/mitbih/raw", help="Directory containing MIT-BIH WFDB files.")
    parser.add_argument("--out", default="ecg_acl/data/mitbih_context_rr/processed", help="Output directory.")
    parser.add_argument("--left", type=int, default=90, help="Samples before R peak at 360 Hz.")
    parser.add_argument("--right", type=int, default=126, help="Samples after R peak at 360 Hz.")
    parser.add_argument("--target-len", type=int, default=216, help="Output beat length.")
    parser.add_argument("--context-radius", type=int, default=1, help="Number of neighbor beats on each side.")
    parser.add_argument("--rr-window", type=int, default=10, help="Half window size for local RR median.")
    parser.add_argument("--max-rr-ratio", type=float, default=3.0, help="Clamp for normalized RR ratio features.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio from DS1 training records.")
    parser.add_argument(
        "--split-mode",
        choices=["record", "beat"],
        default="record",
        help="Use whole-record validation split or beat-level stratified validation split.",
    )
    parser.add_argument(
        "--val-records",
        default=None,
        help="Comma-separated DS1 record ids to hold out for validation. Use semicolons for multiple folds.",
    )
    parser.add_argument("--num-val-folds", type=int, default=1, help="Number of disjoint record-level validation folds.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _extract_beat(lead: np.ndarray, sample: int, left: int, right: int, target_len: int) -> np.ndarray | None:
    start = sample - left
    end = sample + right
    if start < 0 or end > len(lead):
        return None
    beat = _normalize_beat(lead[start:end])
    if len(beat) != target_len:
        beat = resample(beat, target_len).astype(np.float32)
    return beat.astype(np.float32)


def _clip_positive_ratio(value: float, max_ratio: float) -> float:
    return float(np.clip(value, 1.0 / max_ratio, max_ratio))


def _clip_signed_ratio(value: float, max_ratio: float) -> float:
    return float(np.clip(value, -max_ratio, max_ratio))


def _rr_features(
    samples: np.ndarray,
    intervals: np.ndarray,
    idx: int,
    fs: float,
    rr_window: int,
    max_rr_ratio: float,
) -> list[float]:
    prev_rr = max(float(samples[idx] - samples[idx - 1]) / fs, 1e-3)
    next_rr = max(float(samples[idx + 1] - samples[idx]) / fs, 1e-3)

    start = max(0, idx - rr_window)
    stop = min(len(intervals), idx + rr_window + 1)
    local_intervals = intervals[start:stop]
    local_intervals = local_intervals[local_intervals > 0]
    local_rr = float(np.median(local_intervals)) if len(local_intervals) else (prev_rr + next_rr) * 0.5

    positive_intervals = intervals[intervals > 0]
    record_rr = float(np.median(positive_intervals)) if len(positive_intervals) else local_rr
    local_rr = max(local_rr, 1e-3)
    record_rr = max(record_rr, 1e-3)

    return [
        _clip_positive_ratio(prev_rr / local_rr, max_rr_ratio),
        _clip_positive_ratio(next_rr / local_rr, max_rr_ratio),
        _clip_signed_ratio((next_rr - prev_rr) / local_rr, max_rr_ratio),
        _clip_positive_ratio(local_rr / record_rr, max_rr_ratio),
    ]


def _channel_names(context_radius: int) -> list[str]:
    beat_channels = [f"beat_t{offset:+d}" for offset in range(-context_radius, context_radius + 1)]
    return beat_channels + ["rr_prev_ratio", "rr_next_ratio", "rr_delta_ratio", "rr_local_ratio"]


def _load_context_records(
    raw: Path,
    records: set[str],
    left: int,
    right: int,
    target_len: int,
    context_radius: int,
    rr_window: int,
    max_rr_ratio: float,
):
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Install wfdb first: pip install wfdb") from exc

    xs = []
    ys = []
    out_records = []
    out_samples = []
    used_records = []

    for record in _record_ids(raw):
        if record not in records:
            continue

        record_path = str(raw / record)
        signal, fields = wfdb.rdsamp(record_path)
        ann = wfdb.rdann(record_path, "atr")
        fs = float(fields.get("fs", 360.0) or 360.0)
        lead = signal[:, 0]

        beats = []
        for sample, symbol in zip(ann.sample, ann.symbol):
            aami = AAMI_MAP.get(symbol)
            if aami is None:
                continue
            beat = _extract_beat(lead, int(sample), left, right, target_len)
            if beat is None:
                continue
            beats.append({"sample": int(sample), "label": CLASS_TO_ID[aami], "beat": beat})

        if len(beats) <= context_radius * 2:
            continue

        record_added = 0
        samples = np.asarray([item["sample"] for item in beats], dtype=np.float64)
        intervals = np.diff(samples) / fs
        for idx in range(context_radius, len(beats) - context_radius):
            wave_channels = [
                beats[offset_idx]["beat"]
                for offset_idx in range(idx - context_radius, idx + context_radius + 1)
            ]
            rr_values = _rr_features(samples, intervals, idx, fs, rr_window, max_rr_ratio)
            rr_channels = np.repeat(np.asarray(rr_values, dtype=np.float32)[:, None], target_len, axis=1)
            x = np.concatenate([np.stack(wave_channels, axis=0), rr_channels], axis=0)
            xs.append(x.astype(np.float32))
            ys.append(int(beats[idx]["label"]))
            out_records.append(record)
            out_samples.append(int(beats[idx]["sample"]))
            record_added += 1

        if record_added:
            used_records.append(record)

    if not xs:
        raise RuntimeError(f"No context beats were produced from {raw}. Check downloaded records.")

    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(out_records),
        np.asarray(out_samples, dtype=np.int64),
        used_records,
    )


def _counts(y: np.ndarray) -> dict[str, int]:
    return {name: int((y == idx).sum()) for idx, name in enumerate(AAMI_CLASSES)}


def _select_record_group(
    y: np.ndarray,
    records: np.ndarray,
    candidate_records: list[str],
    val_ratio: float,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    record_counts = {
        record: np.bincount(y[records == record], minlength=len(AAMI_CLASSES)).astype(np.float64)
        for record in candidate_records
    }
    total_counts = np.bincount(y, minlength=len(AAMI_CLASSES)).astype(np.float64)
    target_counts = np.maximum(total_counts * val_ratio, 1.0)
    target_samples = max(float(len(y) * val_ratio), 1.0)
    positive_classes = total_counts > 0
    target_record_count = max(1, int(round(len(set(records.tolist())) * val_ratio)))
    candidate_sizes = sorted(
        {
            max(1, min(len(candidate_records), target_record_count + offset))
            for offset in range(-1, 4)
        }
    )

    best_key: tuple[float, float, float, float, float] | None = None
    val_records = []
    for size in candidate_sizes:
        for combo in itertools.combinations(candidate_records, size):
            val_counts = np.sum([record_counts[record] for record in combo], axis=0)
            missing = int(((val_counts == 0) & positive_classes).sum())
            count_error = float(np.mean(((val_counts - target_counts) / target_counts) ** 2))
            sample_error = abs(float(val_counts.sum()) - target_samples) / target_samples
            record_error = abs(size - target_record_count) / max(float(target_record_count), 1.0)
            jitter = float(rng.random() * 1e-6)
            key = (float(missing), count_error, sample_error, record_error, jitter)
            if best_key is None or key < best_key:
                best_key = key
                val_records = list(combo)
    return val_records


def _parse_val_record_groups(value: str | None) -> list[list[str]] | None:
    if not value:
        return None
    groups = []
    for group in value.split(";"):
        records = sorted({item.strip() for item in group.split(",") if item.strip()})
        if records:
            groups.append(records)
    return groups or None


def _record_val_folds(
    y: np.ndarray,
    records: np.ndarray,
    val_ratio: float,
    seed: int,
    num_val_folds: int,
    explicit_val_records: str | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[list[str]]]:
    unique_records = sorted(set(records.tolist()))
    explicit_groups = _parse_val_record_groups(explicit_val_records)
    if explicit_groups:
        val_folds = explicit_groups
        unknown = sorted(set().union(*[set(group) for group in val_folds]) - set(unique_records))
        if unknown:
            raise ValueError(f"Validation records not found in DS1 data: {unknown}")
    else:
        remaining = unique_records[:]
        val_folds = []
        for fold_idx in range(max(1, num_val_folds)):
            if not remaining:
                break
            group = _select_record_group(y, records, remaining, val_ratio, seed + fold_idx)
            if not group:
                break
            val_folds.append(group)
            remaining = [record for record in remaining if record not in set(group)]

    used_val_records = sorted(set().union(*[set(group) for group in val_folds])) if val_folds else []
    val_fold_indices = [np.where(np.isin(records, group))[0] for group in val_folds]
    if not val_fold_indices or not any(len(indices) for indices in val_fold_indices):
        raise RuntimeError("Record-level validation split produced an empty validation set.")
    train_idx = np.where(~np.isin(records, used_val_records))[0]
    if len(train_idx) == 0:
        raise RuntimeError("Record-level validation split used all records and left no training set.")
    return train_idx, val_fold_indices, val_folds


def _record_val_split(
    y: np.ndarray,
    records: np.ndarray,
    val_ratio: float,
    seed: int,
    explicit_val_records: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_idx, val_fold_indices, val_folds = _record_val_folds(
        y,
        records,
        val_ratio,
        seed,
        1,
        explicit_val_records,
    )
    val_idx = val_fold_indices[0]
    val_records = val_folds[0]
    val_mask = np.isin(records, val_records)
    if not val_mask.any():
        raise RuntimeError("Record-level validation split produced an empty validation set.")
    return train_idx, val_idx, val_records


def _save_npz(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    records: np.ndarray,
    samples: np.ndarray,
    channel_names: list[str],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x,
        y=y,
        record=records,
        sample=samples,
        channel_names=np.asarray(channel_names),
    )
    counts = _counts(y)
    print(f"{path}: x={x.shape}, y={y.shape}, counts={counts}")
    return {
        "path": str(path),
        "shape": list(x.shape),
        "counts": counts,
        "records": dict(Counter(records.tolist())),
    }


def main() -> None:
    args = parse_args()
    raw = Path(args.raw)
    out = Path(args.out)
    if not raw.exists():
        raise FileNotFoundError(f"Raw MIT-BIH directory not found: {raw}")
    if args.context_radius < 1:
        raise ValueError("--context-radius must be >= 1")

    channel_names = _channel_names(args.context_radius)
    x_train_all, y_train_all, train_records_all, train_samples_all, train_records = _load_context_records(
        raw,
        DS1_RECORDS,
        args.left,
        args.right,
        args.target_len,
        args.context_radius,
        args.rr_window,
        args.max_rr_ratio,
    )
    x_test, y_test, test_records_all, test_samples_all, test_records = _load_context_records(
        raw,
        DS2_RECORDS,
        args.left,
        args.right,
        args.target_len,
        args.context_radius,
        args.rr_window,
        args.max_rr_ratio,
    )
    val_fold_indices: list[np.ndarray] = []
    val_folds: list[list[str]] = []
    if args.split_mode == "record":
        train_idx, val_fold_indices, val_folds = _record_val_folds(
            y_train_all,
            train_records_all,
            args.val_ratio,
            args.seed,
            args.num_val_folds,
            args.val_records,
        )
        val_idx = np.concatenate(val_fold_indices)
        val_records = sorted(set().union(*[set(group) for group in val_folds]))
    else:
        train_idx, val_idx = _stratified_val_split(y_train_all, args.val_ratio, args.seed)
        val_records = sorted(set(train_records_all[val_idx].tolist()))
        val_fold_indices = [val_idx]
        val_folds = [val_records]

    val_fold_summaries = []
    for fold_idx, fold_indices in enumerate(val_fold_indices, start=1):
        val_fold_summaries.append(
            _save_npz(
                out / f"val_fold_{fold_idx:02d}.npz",
                x_train_all[fold_indices],
                y_train_all[fold_indices],
                train_records_all[fold_indices],
                train_samples_all[fold_indices],
                channel_names,
            )
        )

    summary = {
        "class_names": AAMI_CLASSES,
        "channel_names": channel_names,
        "context_radius": args.context_radius,
        "rr_window": args.rr_window,
        "max_rr_ratio": args.max_rr_ratio,
        "split_mode": args.split_mode,
        "val_ratio": args.val_ratio,
        "num_val_folds": len(val_fold_indices),
        "val_records": val_records,
        "val_folds": val_folds,
        "val_fold_npzs": [str(out / f"val_fold_{idx:02d}.npz") for idx in range(1, len(val_fold_indices) + 1)],
        "input_channels": len(channel_names),
        "input_length": args.target_len,
        "splits": {
            "train": _save_npz(
                out / "train.npz",
                x_train_all[train_idx],
                y_train_all[train_idx],
                train_records_all[train_idx],
                train_samples_all[train_idx],
                channel_names,
            ),
            "val": _save_npz(
                out / "val.npz",
                x_train_all[val_idx],
                y_train_all[val_idx],
                train_records_all[val_idx],
                train_samples_all[val_idx],
                channel_names,
            ),
            "test": _save_npz(out / "test.npz", x_test, y_test, test_records_all, test_samples_all, channel_names),
        },
        "val_fold_summaries": val_fold_summaries,
        "train_records": train_records,
        "test_records": test_records,
    }
    (out / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"metadata: {out / 'metadata.json'}")


if __name__ == "__main__":
    main()
