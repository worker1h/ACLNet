from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGNpzDataset(Dataset):
    """Dataset for preprocessed ECG beat windows stored as npz files."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessed dataset not found: {path}")
        arrays = np.load(path)
        self.x = arrays["x"].astype(np.float32)
        self.y = arrays["y"].astype(np.int64)
        if self.x.ndim == 2:
            self.x = self.x[:, None, :]
        if self.x.ndim != 3:
            raise ValueError(f"Expected x with shape [N, C, T], got {self.x.shape}")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must contain the same number of samples")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)

    @property
    def labels(self) -> np.ndarray:
        return self.y


class SyntheticECGDataset(Dataset):
    """Small deterministic dataset for smoke-testing the training pipeline."""

    def __init__(
        self,
        size: int = 256,
        num_classes: int = 5,
        channels: int = 1,
        length: int = 216,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, length, dtype=np.float32)
        xs = []
        ys = []
        for idx in range(size):
            label = idx % num_classes
            freq = 3.0 + label * 0.8
            phase = rng.uniform(0, 2 * np.pi)
            wave = np.sin(2 * np.pi * freq * t + phase)
            wave += 0.35 * np.exp(-((t - (0.35 + 0.04 * label)) ** 2) / 0.0015)
            wave += 0.08 * rng.normal(size=length)
            xs.append(np.tile(wave[None, :], (channels, 1)))
            ys.append(label)
        self.x = np.stack(xs).astype(np.float32)
        self.y = np.asarray(ys, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)

    @property
    def labels(self) -> np.ndarray:
        return self.y


class ECGAugmentDataset(Dataset):
    """Applies lightweight beat-level augmentation, optionally only to minority classes."""

    def __init__(
        self,
        dataset: Dataset,
        augment_classes: set[int] | None,
        noise_std: float = 0.02,
        scale_std: float = 0.08,
        max_shift: int = 8,
        waveform_channels: int | None = None,
    ):
        self.dataset = dataset
        self.augment_classes = augment_classes
        self.noise_std = noise_std
        self.scale_std = scale_std
        self.max_shift = max_shift
        self.waveform_channels = waveform_channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        label = int(y.item())
        if self.augment_classes is None or label in self.augment_classes:
            x = x.clone()
            waveform_channels = x.shape[0] if self.waveform_channels is None else min(self.waveform_channels, x.shape[0])
            wave = x[:waveform_channels]
            if self.max_shift > 0:
                shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
                wave = torch.roll(wave, shifts=shift, dims=-1)
            if self.scale_std > 0:
                scale = 1.0 + torch.randn((), dtype=x.dtype) * self.scale_std
                wave = wave * scale
            if self.noise_std > 0:
                wave = wave + torch.randn_like(wave) * self.noise_std
            x[:waveform_channels] = wave
        return x, y

    @property
    def labels(self) -> np.ndarray:
        if hasattr(self.dataset, "labels"):
            return np.asarray(self.dataset.labels, dtype=np.int64)
        return np.asarray([int(self.dataset[idx][1]) for idx in range(len(self.dataset))], dtype=np.int64)
