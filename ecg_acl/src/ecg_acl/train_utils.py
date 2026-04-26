from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, random_split
from torch.utils.data import Subset

from .data import ECGAugmentDataset, ECGNpzDataset, SyntheticECGDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class DynamicNormalDownsampleSampler(Sampler):
    """Randomly downsample the normal class each epoch while keeping abnormal samples available."""

    def __init__(
        self,
        labels: np.ndarray,
        normal_class: int = 0,
        ratio_to_abnormal: float = 3.0,
        sampler_power: float = 1.0,
        weighted: bool = True,
        replacement: bool = True,
        num_samples: str | int | None = "pool",
        seed: int = 42,
        mode: str = "dynamic",
        num_classes: int | None = None,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.normal_class = int(normal_class)
        self.ratio_to_abnormal = float(ratio_to_abnormal)
        self.sampler_power = float(sampler_power)
        self.weighted = bool(weighted)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.mode = str(mode).lower()
        self.num_classes = int(num_classes or (self.labels.max() + 1))
        self.epoch = 0

        if self.labels.ndim != 1:
            raise ValueError(f"labels must be one-dimensional, got shape {self.labels.shape}")
        if self.ratio_to_abnormal <= 0:
            raise ValueError("normal_downsample.ratio_to_abnormal must be > 0")
        if self.mode not in {"dynamic", "static"}:
            raise ValueError("normal_downsample.mode must be 'dynamic' or 'static'")

        self.normal_indices = np.flatnonzero(self.labels == self.normal_class)
        self.abnormal_indices = np.flatnonzero(self.labels != self.normal_class)
        if len(self.abnormal_indices) == 0:
            raise ValueError("normal downsampling requires at least one non-normal training sample")

        requested_normal = int(round(len(self.abnormal_indices) * self.ratio_to_abnormal))
        self.normal_target = min(len(self.normal_indices), max(1, requested_normal))
        self.pool_size = int(self.normal_target + len(self.abnormal_indices))
        self.num_samples = self._resolve_num_samples(num_samples)

    def _resolve_num_samples(self, num_samples: str | int | None) -> int:
        if num_samples is None:
            return self.pool_size
        if isinstance(num_samples, str):
            value = num_samples.strip().lower()
            if value == "pool":
                return self.pool_size
            if value == "full":
                return len(self.labels)
            return max(1, int(value))
        return max(1, int(num_samples))

    def __iter__(self):
        seed = self.seed + self.epoch if self.mode == "dynamic" else self.seed
        rng = np.random.default_rng(seed)
        if self.normal_target >= len(self.normal_indices):
            normal_subset = self.normal_indices
        else:
            normal_subset = rng.choice(self.normal_indices, size=self.normal_target, replace=False)
        pool_indices = np.concatenate([normal_subset, self.abnormal_indices]).astype(np.int64)

        if self.weighted:
            pool_labels = self.labels[pool_indices]
            counts = np.bincount(pool_labels, minlength=self.num_classes).clip(min=1)
            weights = 1.0 / np.power(counts[pool_labels], self.sampler_power)
            probabilities = weights / weights.sum()
            sample_count = self.num_samples
            if not self.replacement:
                sample_count = min(sample_count, len(pool_indices))
            sampled = rng.choice(
                pool_indices,
                size=sample_count,
                replace=self.replacement,
                p=probabilities,
            )
        else:
            sampled = rng.permutation(pool_indices)

        if self.mode == "dynamic":
            self.epoch += 1
        return iter(sampled.astype(np.int64).tolist())

    def __len__(self) -> int:
        return self.num_samples if self.weighted else self.pool_size

    def summary(self) -> dict[str, Any]:
        original_counts = np.bincount(self.labels, minlength=self.num_classes).astype(int)
        pool_counts = original_counts.copy()
        pool_counts[self.normal_class] = self.normal_target
        return {
            "type": "dynamic_normal_downsample",
            "mode": self.mode,
            "normal_class": self.normal_class,
            "ratio_to_abnormal": self.ratio_to_abnormal,
            "normal_total": int(len(self.normal_indices)),
            "abnormal_total": int(len(self.abnormal_indices)),
            "normal_target": int(self.normal_target),
            "pool_size": int(self.pool_size),
            "num_samples": int(len(self)),
            "weighted": self.weighted,
            "replacement": self.replacement,
            "sampler_power": self.sampler_power,
            "original_class_counts": original_counts.tolist(),
            "pool_class_counts": pool_counts.tolist(),
        }


def build_loaders(cfg: dict[str, Any], synthetic: bool, batch_size: int | None = None):
    batch_size = batch_size or int(cfg["train"]["batch_size"])
    workers = int(cfg["train"].get("num_workers", 0))
    if synthetic:
        data_cfg = cfg["data"]
        dataset = SyntheticECGDataset(
            size=384,
            num_classes=int(data_cfg["num_classes"]),
            channels=int(data_cfg["input_channels"]),
            length=int(data_cfg["input_length"]),
            seed=int(cfg["seed"]),
        )
        train_ds, val_ds, test_ds = random_split(
            dataset,
            [256, 64, 64],
            generator=torch.Generator().manual_seed(int(cfg["seed"])),
        )
    else:
        data_cfg = cfg["data"]
        train_ds = ECGNpzDataset(data_cfg["train_npz"])
        val_ds = ECGNpzDataset(data_cfg["val_npz"])
        test_ds = ECGNpzDataset(data_cfg["test_npz"])

    aug_cfg = cfg.get("augmentation", {})
    if bool(aug_cfg.get("enabled", False)) and not synthetic:
        train_labels = get_dataset_labels(train_ds)
        counts = np.bincount(train_labels, minlength=int(cfg["data"]["num_classes"]))
        if bool(aug_cfg.get("minority_only", True)):
            max_count = counts.max()
            threshold = max(1, int(max_count * float(aug_cfg.get("minority_max_fraction", 0.25))))
            augment_classes = {int(idx) for idx, count in enumerate(counts) if 0 < count <= threshold}
        else:
            augment_classes = None
        train_ds = ECGAugmentDataset(
            train_ds,
            augment_classes=augment_classes,
            noise_std=float(aug_cfg.get("noise_std", 0.02)),
            scale_std=float(aug_cfg.get("scale_std", 0.08)),
            max_shift=int(aug_cfg.get("max_shift", 8)),
            waveform_channels=(
                int(cfg.get("data", {}).get("waveform_channels"))
                if cfg.get("data", {}).get("waveform_channels") is not None
                else None
            ),
        )

    sampler = None
    shuffle = True
    imbalance_cfg = cfg.get("imbalance", {})
    normal_downsample_cfg = imbalance_cfg.get("normal_downsample", {}) or {}
    if bool(normal_downsample_cfg.get("enabled", False)) and not synthetic:
        train_labels = get_dataset_labels(train_ds)
        sampler = DynamicNormalDownsampleSampler(
            labels=train_labels,
            normal_class=int(normal_downsample_cfg.get("normal_class", 0)),
            ratio_to_abnormal=float(normal_downsample_cfg.get("ratio_to_abnormal", 3.0)),
            sampler_power=float(imbalance_cfg.get("sampler_power", 1.0)),
            weighted=bool(normal_downsample_cfg.get("weighted", True)),
            replacement=bool(normal_downsample_cfg.get("replacement", True)),
            num_samples=normal_downsample_cfg.get("num_samples", "pool"),
            seed=int(cfg["seed"]),
            mode=str(normal_downsample_cfg.get("mode", "dynamic")),
            num_classes=int(cfg["data"]["num_classes"]),
        )
        shuffle = False
    elif bool(imbalance_cfg.get("weighted_sampler", False)):
        train_labels = get_dataset_labels(train_ds)
        class_counts = np.bincount(train_labels, minlength=int(cfg["data"]["num_classes"])).clip(min=1)
        sampler_power = float(imbalance_cfg.get("sampler_power", 1.0))
        sample_weights = 1.0 / np.power(class_counts[train_labels], sampler_power)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader


def get_dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    if isinstance(dataset, Subset):
        labels = get_dataset_labels(dataset.dataset)
        return labels[np.asarray(dataset.indices, dtype=np.int64)]
    return np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))], dtype=np.int64)


def compute_class_weights(labels: np.ndarray, num_classes: int, beta: float, max_weight: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.clip(effective_num, 1e-12, None)
    weights = weights / weights.mean()
    weights = np.clip(weights, 1.0 / max_weight, max_weight)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
