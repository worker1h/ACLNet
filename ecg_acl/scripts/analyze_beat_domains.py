from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft, welch


AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_NAMES = {
    "N": "Normal",
    "S": "Supraventricular",
    "V": "Ventricular",
    "F": "Fusion",
    "Q": "Unknown/Paced",
}
COLORS = {
    "N": "#2f6fbb",
    "S": "#d7812a",
    "V": "#2f8f5b",
    "F": "#a33f3f",
    "Q": "#7a5ab8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Class-wise ECG time/frequency/time-frequency analysis.")
    parser.add_argument("--data-dir", default="ecg_acl/data/mitbih/processed")
    parser.add_argument("--out-dir", default="ecg_acl/analysis_outputs/class_domain_analysis")
    parser.add_argument("--fs", type=float, default=360.0)
    parser.add_argument("--r-index", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def load_splits(data_dir: Path):
    xs = []
    ys = []
    split_names = []
    split_counts = {}
    for split in ["train", "val", "test"]:
        arrays = np.load(data_dir / f"{split}.npz")
        x = arrays["x"].astype(np.float32)
        y = arrays["y"].astype(np.int64)
        if x.ndim == 3:
            x = x[:, 0, :]
        xs.append(x)
        ys.append(y)
        split_names.extend([split] * len(y))
        split_counts[split] = {AAMI_CLASSES[k]: int(v) for k, v in Counter(y.tolist()).items()}
    return np.concatenate(xs), np.concatenate(ys), np.asarray(split_names), split_counts


def robust_preprocess(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32).copy()
    y = y - np.median(y, axis=1, keepdims=True)
    scale = np.percentile(np.abs(y), 95, axis=1, keepdims=True)
    std = y.std(axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, std + 1e-6, scale)
    return y / scale


def bandpower(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freq >= low) & (freq < high)
    if not mask.any():
        return np.zeros(psd.shape[0], dtype=np.float32)
    return np.trapezoid(psd[:, mask], freq[mask], axis=1)


def compute_features(x: np.ndarray, fs: float, r_index: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    freq, psd = welch(x, fs=fs, nperseg=min(128, x.shape[1]), noverlap=min(64, x.shape[1] // 2), axis=1)
    total_power = np.trapezoid(psd, freq, axis=1) + 1e-12
    low = bandpower(freq, psd, 0.5, 5.0)
    mid = bandpower(freq, psd, 5.0, 15.0)
    high = bandpower(freq, psd, 15.0, 40.0)
    upper = bandpower(freq, psd, 40.0, 100.0)
    centroid = (psd * freq[None, :]).sum(axis=1) / (psd.sum(axis=1) + 1e-12)
    rows = []
    for idx in range(x.shape[0]):
        beat = x[idx]
        diff = np.diff(beat)
        rows.append(
            {
                "r_amp": float(beat[min(r_index, len(beat) - 1)]),
                "max_amp": float(beat.max()),
                "min_amp": float(beat.min()),
                "peak_to_peak": float(beat.max() - beat.min()),
                "energy": float(np.mean(beat**2)),
                "mean_abs_slope": float(np.mean(np.abs(diff))),
                "zero_crossings": int(np.count_nonzero(np.diff(np.signbit(beat)))),
                "spectral_centroid_hz": float(centroid[idx]),
                "bandpower_0p5_5": float(low[idx] / total_power[idx]),
                "bandpower_5_15": float(mid[idx] / total_power[idx]),
                "bandpower_15_40": float(high[idx] / total_power[idx]),
                "bandpower_40_100": float(upper[idx] / total_power[idx]),
            }
        )
    return rows, freq, psd


def summarize_features(rows: list[dict], y: np.ndarray, out_csv: Path) -> None:
    feature_names = [name for name in rows[0] if name != "class"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "feature", "mean", "std", "median", "q25", "q75"])
        for class_id, label in enumerate(AAMI_CLASSES):
            indices = np.where(y == class_id)[0]
            for feature in feature_names:
                values = np.asarray([rows[idx][feature] for idx in indices], dtype=np.float64)
                writer.writerow(
                    [
                        label,
                        feature,
                        float(values.mean()),
                        float(values.std()),
                        float(np.median(values)),
                        float(np.percentile(values, 25)),
                        float(np.percentile(values, 75)),
                    ]
                )


def plot_counts(y: np.ndarray, split_names: np.ndarray, out_path: Path) -> None:
    x = np.arange(len(AAMI_CLASSES))
    bottoms = np.zeros(len(AAMI_CLASSES))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    for split, color in [("train", "#5c83b6"), ("val", "#d8a03d"), ("test", "#6ba56f")]:
        counts = [int(((y == idx) & (split_names == split)).sum()) for idx in range(len(AAMI_CLASSES))]
        ax.bar(x, counts, bottom=bottoms, label=split, color=color)
        bottoms += counts
    ax.set_xticks(x, [f"{label}\n{CLASS_NAMES[label]}" for label in AAMI_CLASSES])
    ax.set_ylabel("Beats")
    ax.set_title("Class Distribution Across Splits")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_time_mean(x: np.ndarray, y: np.ndarray, fs: float, r_index: int, out_path: Path) -> None:
    t = (np.arange(x.shape[1]) - r_index) / fs
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    for class_id, label in enumerate(AAMI_CLASSES):
        class_x = x[y == class_id]
        mean = class_x.mean(axis=0)
        std = class_x.std(axis=0)
        color = COLORS[label]
        ax.plot(t, mean, label=f"{label} {CLASS_NAMES[label]}", color=color, lw=2)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.axvline(0.0, color="#222222", lw=1, ls="--", alpha=0.8)
    ax.set_title("Time Domain: Class Mean Waveforms With 1 SD Envelope")
    ax.set_xlabel("Time from annotated R peak (s)")
    ax.set_ylabel("Robust-normalized amplitude")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_examples(x: np.ndarray, y: np.ndarray, fs: float, r_index: int, out_path: Path, seed: int, max_examples: int) -> None:
    rng = np.random.default_rng(seed)
    t = (np.arange(x.shape[1]) - r_index) / fs
    fig, axes = plt.subplots(len(AAMI_CLASSES), 1, figsize=(10, 9), dpi=180, sharex=True, sharey=True)
    for class_id, label in enumerate(AAMI_CLASSES):
        ax = axes[class_id]
        indices = np.where(y == class_id)[0]
        chosen = rng.choice(indices, size=min(max_examples, len(indices)), replace=False)
        for idx in chosen:
            ax.plot(t, x[idx], color=COLORS[label], alpha=0.18, lw=0.8)
        ax.plot(t, x[indices].mean(axis=0), color="#111111", lw=1.5)
        ax.axvline(0.0, color="#222222", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
    axes[0].set_title("Time Domain: Random Beat Examples Per Class")
    axes[-1].set_xlabel("Time from annotated R peak (s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_psd(freq: np.ndarray, psd: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    mask = freq <= 100.0
    for class_id, label in enumerate(AAMI_CLASSES):
        class_psd = psd[y == class_id]
        mean = class_psd.mean(axis=0)
        q25, q75 = np.percentile(class_psd, [25, 75], axis=0)
        color = COLORS[label]
        ax.plot(freq[mask], 10 * np.log10(mean[mask] + 1e-12), color=color, lw=2, label=f"{label} {CLASS_NAMES[label]}")
        ax.fill_between(
            freq[mask],
            10 * np.log10(q25[mask] + 1e-12),
            10 * np.log10(q75[mask] + 1e-12),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    ax.set_title("Frequency Domain: Mean Welch PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (dB)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def average_stft(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    specs = []
    freq = None
    time = None
    for beat in x:
        freq, time, zxx = stft(beat, fs=fs, nperseg=48, noverlap=42, boundary=None)
        specs.append(np.abs(zxx))
    return freq, time, np.mean(np.stack(specs), axis=0)


def plot_stft_grid(x: np.ndarray, y: np.ndarray, fs: float, r_index: int, out_path: Path) -> None:
    averages = {}
    vmax = -np.inf
    for class_id, label in enumerate(AAMI_CLASSES):
        freq, time, spec = average_stft(x[y == class_id], fs)
        mask = freq <= 100.0
        db = 20 * np.log10(spec[mask] + 1e-6)
        averages[label] = (freq[mask], time - (r_index / fs), db)
        vmax = max(vmax, float(np.percentile(db, 99)))
    vmin = vmax - 45
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), dpi=180, sharex=True, sharey=True)
    axes = axes.ravel()
    im = None
    for ax, label in zip(axes, AAMI_CLASSES):
        freq_i, time_i, db = averages[label]
        im = ax.pcolormesh(time_i, freq_i, db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"{label} {CLASS_NAMES[label]}")
        ax.axvline(0.0, color="#ffffff", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(0, 100)
    axes[-1].axis("off")
    for ax in axes[:3]:
        ax.set_xlabel("")
    for ax in axes[::3]:
        ax.set_ylabel("Frequency (Hz)")
    for ax in axes[3:]:
        ax.set_xlabel("Time from R peak (s)")
    fig.suptitle("Time-Frequency Domain: Average STFT Magnitude", y=0.99)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:-1], fraction=0.025, pad=0.02)
        cbar.set_label("Magnitude (dB)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_boxplots(rows: list[dict], y: np.ndarray, out_path: Path) -> None:
    feature_names = [
        "peak_to_peak",
        "energy",
        "mean_abs_slope",
        "spectral_centroid_hz",
        "bandpower_5_15",
        "bandpower_15_40",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=180)
    axes = axes.ravel()
    for ax, feature in zip(axes, feature_names):
        data = [
            np.asarray([rows[idx][feature] for idx in np.where(y == class_id)[0]], dtype=np.float64)
            for class_id in range(len(AAMI_CLASSES))
        ]
        box = ax.boxplot(data, labels=AAMI_CLASSES, patch_artist=True, showfliers=False)
        for patch, label in zip(box["boxes"], AAMI_CLASSES):
            patch.set_facecolor(COLORS[label])
            patch.set_alpha(0.45)
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Class-wise Time/Frequency Feature Distributions", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_notes(out_path: Path, counts: dict, summary_csv: Path) -> None:
    lines = [
        "# ECG Class-wise Domain Analysis",
        "",
        "AAMI class labels:",
        "",
    ]
    for label in AAMI_CLASSES:
        lines.append(f"- `{label}`: {CLASS_NAMES[label]}")
    lines.extend(
        [
            "",
            "Preprocessing used for this analysis: per-beat median baseline removal followed by robust scaling",
            "with the 95th percentile absolute amplitude. This keeps morphology comparable without using",
            "class labels to change the signal shape.",
            "",
            "Split counts:",
            "",
            "```json",
            json.dumps(counts, indent=2),
            "```",
            "",
            f"Feature summary CSV: `{summary_csv.name}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y, split_names, split_counts = load_splits(data_dir)
    x = robust_preprocess(x_raw)
    np.savez_compressed(
        out_dir / "preprocessed_by_class.npz",
        **{label: x[y == idx].astype(np.float32) for idx, label in enumerate(AAMI_CLASSES)},
    )

    rows, freq, psd = compute_features(x, args.fs, args.r_index)
    summary_csv = out_dir / "domain_feature_summary.csv"
    summarize_features(rows, y, summary_csv)

    plot_counts(y, split_names, out_dir / "01_class_counts.png")
    plot_time_mean(x, y, args.fs, args.r_index, out_dir / "02_time_domain_mean_std.png")
    plot_examples(x, y, args.fs, args.r_index, out_dir / "03_time_domain_examples.png", args.seed, args.max_examples)
    plot_psd(freq, psd, y, out_dir / "04_frequency_domain_psd.png")
    plot_stft_grid(x, y, args.fs, args.r_index, out_dir / "05_time_frequency_stft.png")
    plot_feature_boxplots(rows, y, out_dir / "06_feature_boxplots.png")
    write_notes(out_dir / "analysis_notes.md", split_counts, summary_csv)

    class_counts = {label: int((y == idx).sum()) for idx, label in enumerate(AAMI_CLASSES)}
    print(f"out_dir={out_dir.resolve()}")
    print(f"class_counts={class_counts}")
    print("generated=01_class_counts.png,02_time_domain_mean_std.png,03_time_domain_examples.png,")
    print("generated=04_frequency_domain_psd.png,05_time_frequency_stft.png,06_feature_boxplots.png")


if __name__ == "__main__":
    main()
