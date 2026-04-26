from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
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
    if bool(cfg.get("imbalance", {}).get("weighted_sampler", False)):
        train_labels = get_dataset_labels(train_ds)
        class_counts = np.bincount(train_labels, minlength=int(cfg["data"]["num_classes"])).clip(min=1)
        sampler_power = float(cfg.get("imbalance", {}).get("sampler_power", 1.0))
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
