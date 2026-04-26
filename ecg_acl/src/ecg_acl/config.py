from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "train_npz": "ecg_acl/data/mitbih/processed/train.npz",
        "val_npz": "ecg_acl/data/mitbih/processed/val.npz",
        "val_npzs": None,
        "test_npz": "ecg_acl/data/mitbih/processed/test.npz",
        "num_classes": 5,
        "input_channels": 1,
        "waveform_channels": None,
        "input_length": 216,
    },
    "model": {
        "name": "resnet1d",
        "base_channels": 64,
        "embedding_dim": 256,
        "branch_dim": 128,
        "birnn_hidden": 64,
        "birnn_layers": 2,
        "dropout": 0.3,
        "branches": ["birnn", "efficientnet", "sequential", "lenet"],
        "aux_weight": 0.0,
    },
    "train": {
        "epochs": 80,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "num_workers": 0,
        "device": "auto",
        "selection_metric": "macro_f1",
        "selection_metrics": ["macro_f1"],
        "min_selection_epoch": 1,
        "composite_metrics": {},
        "save_top_k": 1,
        "minority_classes": [1, 2, 3, 4],
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "clean_topk": True,
    },
    "scheduler": {
        "name": "cosine",
        "min_lr": 1e-5,
        "factor": 0.5,
        "patience": 8,
        "step_size": 20,
        "gamma": 0.5,
    },
    "imbalance": {
        "weighted_sampler": True,
        "sampler_power": 1.0,
        "loss_type": "class_balanced_ce",
        "class_balanced_loss": True,
        "effective_num_beta": 0.999,
        "max_class_weight": 8.0,
        "focal_gamma": 2.0,
    },
    "augmentation": {
        "enabled": False,
        "minority_only": True,
        "minority_max_fraction": 0.25,
        "noise_std": 0.02,
        "scale_std": 0.08,
        "max_shift": 8,
    },
    "affinity": {
        "start_epoch": 20,
        "top_k": 5,
        "overlap_threshold": 2,
        "inter_weight": 0.1,
        "intra_weight": 0.1,
        "temperature": 0.1,
        "margin": 0.1,
        "memory_size": 1024,
    },
    "output": {"work_dir": "ecg_acl/work_dirs/mitbih_resnet1d_acl", "log_file": "train.log"},
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            key, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"Unsupported config line: {raw_line.rstrip()}")
            while indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            key = key.strip()
            value = value.strip()
            if value == "":
                child: dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_scalar(value)
    return root


def load_config(path: str | None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except ImportError:
        loaded = _load_simple_yaml(config_path)
    return _deep_update(cfg, loaded)
