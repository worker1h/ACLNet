from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample


# AAMI EC57 superclasses commonly used for MIT-BIH beat classification.
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q", "P": "Q", "|": "Q",
}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(AAMI_CLASSES)}

# Patient-oriented split used in many MIT-BIH experiments.
DS1_RECORDS = {
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
}
DS2_RECORDS = {
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MIT-BIH into ECG beat npz files.")
    parser.add_argument("--raw", default="ecg_acl/data/mitbih/raw", help="Directory containing MIT-BIH WFDB files.")
    parser.add_argument("--out", default="ecg_acl/data/mitbih/processed", help="Output directory.")
    parser.add_argument("--left", type=int, default=90, help="Samples before R peak at 360 Hz.")
    parser.add_argument("--right", type=int, default=126, help="Samples after R peak at 360 Hz.")
    parser.add_argument("--target-len", type=int, default=216, help="Output beat length.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio from DS1 training records.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _record_ids(raw: Path) -> list[str]:
    return sorted(path.stem for path in raw.glob("*.hea"))


def _normalize_beat(beat: np.ndarray) -> np.ndarray:
    beat = beat.astype(np.float32)
    beat = beat - np.median(beat)
    scale = np.percentile(np.abs(beat), 95)
    if scale < 1e-6:
        scale = np.std(beat) + 1e-6
    return beat / scale


def _load_records(raw: Path, records: set[str], left: int, right: int, target_len: int):
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Install wfdb first: pip install wfdb") from exc

    xs = []
    ys = []
    used_records = []
    for record in _record_ids(raw):
        if record not in records:
            continue
        record_path = str(raw / record)
        signal, _ = wfdb.rdsamp(record_path)
        ann = wfdb.rdann(record_path, "atr")
        lead = signal[:, 0]
        for sample, symbol in zip(ann.sample, ann.symbol):
            aami = AAMI_MAP.get(symbol)
            if aami is None:
                continue
            start = sample - left
            end = sample + right
            if start < 0 or end > len(lead):
                continue
            beat = _normalize_beat(lead[start:end])
            if len(beat) != target_len:
                beat = resample(beat, target_len).astype(np.float32)
            xs.append(beat[None, :])
            ys.append(CLASS_TO_ID[aami])
        used_records.append(record)
    if not xs:
        raise RuntimeError(f"No beats were produced from {raw}. Check downloaded records.")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), used_records


def _stratified_val_split(y: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in np.unique(y):
        indices = np.where(y == cls)[0]
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 0
        val_idx.extend(indices[:n_val].tolist())
        train_idx.extend(indices[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx), np.asarray(val_idx)


def _save_npz(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y)
    counts = {AAMI_CLASSES[k]: int(v) for k, v in Counter(y.tolist()).items()}
    print(f"{path}: x={x.shape}, y={y.shape}, counts={counts}")


def main() -> None:
    args = parse_args()
    raw = Path(args.raw)
    out = Path(args.out)
    if not raw.exists():
        raise FileNotFoundError(f"Raw MIT-BIH directory not found: {raw}")

    x_train_all, y_train_all, train_records = _load_records(
        raw, DS1_RECORDS, args.left, args.right, args.target_len
    )
    x_test, y_test, test_records = _load_records(raw, DS2_RECORDS, args.left, args.right, args.target_len)
    train_idx, val_idx = _stratified_val_split(y_train_all, args.val_ratio, args.seed)

    _save_npz(out / "train.npz", x_train_all[train_idx], y_train_all[train_idx])
    _save_npz(out / "val.npz", x_train_all[val_idx], y_train_all[val_idx])
    _save_npz(out / "test.npz", x_test, y_test)
    print(f"train records: {train_records}")
    print(f"test records: {test_records}")


if __name__ == "__main__":
    main()
