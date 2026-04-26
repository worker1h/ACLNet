from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ECGResNet1D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        base_channels: int = 64,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels),
            ResidualBlock1D(base_channels, base_channels * 2, stride=2),
            ResidualBlock1D(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock1D(base_channels * 4, base_channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Linear(base_channels * 4, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1)
        embedding = self.embedding(x)
        logits = self.classifier(F.relu(embedding))
        return logits, embedding


class MBConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 4):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return F.silu(out)


class EfficientNet1DBranch(nn.Module):
    def __init__(self, input_channels: int, branch_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            MBConv1D(32, 16, stride=1, expansion=2),
            MBConv1D(16, 24, stride=2, expansion=4),
            MBConv1D(24, 40, stride=2, expansion=4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(40, branch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return F.relu(self.proj(x))


class SequentialNet1DBranch(nn.Module):
    def __init__(self, input_channels: int, branch_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(32, branch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return F.relu(self.proj(x))


class LeNet1DBranch(nn.Module):
    def __init__(self, input_channels: int, branch_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 6, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(16, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, branch_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return F.relu(self.proj(x))


class BiGRUBranch(nn.Module):
    def __init__(
        self,
        input_channels: int,
        branch_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, branch_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        if x.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                _, hidden = self.gru(x)
        else:
            _, hidden = self.gru(x)
        last_forward = hidden[-2]
        last_backward = hidden[-1]
        return self.proj(torch.cat([last_forward, last_backward], dim=1))


class ECGHybridFusionNet(nn.Module):
    """BiGRU + PHM-style lightweight CNN branches with a fused ACL embedding."""

    _BRANCH_BUILDERS = {
        "birnn": BiGRUBranch,
        "efficientnet": EfficientNet1DBranch,
        "sequential": SequentialNet1DBranch,
        "lenet": LeNet1DBranch,
    }

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        embedding_dim: int = 256,
        branch_dim: int = 128,
        birnn_hidden: int = 64,
        birnn_layers: int = 2,
        dropout: float = 0.3,
        branches: list[str] | None = None,
    ):
        super().__init__()
        self.branch_names = branches or ["birnn", "efficientnet", "sequential", "lenet"]
        unknown = sorted(set(self.branch_names) - set(self._BRANCH_BUILDERS))
        if unknown:
            raise ValueError(f"Unknown fusion branches: {unknown}")

        modules = {}
        for name in self.branch_names:
            if name == "birnn":
                modules[name] = BiGRUBranch(
                    input_channels=input_channels,
                    branch_dim=branch_dim,
                    hidden_size=birnn_hidden,
                    num_layers=birnn_layers,
                    dropout=dropout,
                )
            else:
                modules[name] = self._BRANCH_BUILDERS[name](input_channels, branch_dim)
        self.branches = nn.ModuleDict(modules)
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * len(self.branch_names), len(self.branch_names)),
            nn.Softmax(dim=1),
        )
        self.embedding = nn.Sequential(
            nn.Linear(branch_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.aux_heads = nn.ModuleDict({name: nn.Linear(branch_dim, num_classes) for name in self.branch_names})

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        branch_features = [self.branches[name](x) for name in self.branch_names]
        stacked = torch.stack(branch_features, dim=1)
        gate_input = torch.cat(branch_features, dim=1)
        weights = self.gate(gate_input).unsqueeze(-1)
        fused = (stacked * weights).sum(dim=1)
        embedding = self.embedding(fused)
        logits = self.classifier(embedding)
        aux_logits = {
            name: self.aux_heads[name](feature)
            for name, feature in zip(self.branch_names, branch_features)
        }
        return logits, embedding, aux_logits


def build_ecg_model(cfg: dict) -> nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "resnet1d")).lower()
    if model_name == "resnet1d":
        return ECGResNet1D(
            input_channels=int(cfg["data"]["input_channels"]),
            num_classes=int(cfg["data"]["num_classes"]),
            base_channels=int(model_cfg.get("base_channels", 64)),
            embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        )
    if model_name in {"fusion", "hybrid_fusion", "birnn_phm_acl"}:
        return ECGHybridFusionNet(
            input_channels=int(cfg["data"]["input_channels"]),
            num_classes=int(cfg["data"]["num_classes"]),
            embedding_dim=int(model_cfg.get("embedding_dim", 256)),
            branch_dim=int(model_cfg.get("branch_dim", 128)),
            birnn_hidden=int(model_cfg.get("birnn_hidden", 64)),
            birnn_layers=int(model_cfg.get("birnn_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            branches=list(model_cfg.get("branches", ["birnn", "efficientnet", "sequential", "lenet"])),
        )
    raise ValueError(f"Unsupported model.name: {model_name}")


def unpack_model_output(output):
    if isinstance(output, dict):
        return output["logits"], output["embedding"], output.get("aux_logits", {})
    if len(output) == 2:
        logits, embedding = output
        return logits, embedding, {}
    if len(output) == 3:
        logits, embedding, aux_logits = output
        return logits, embedding, aux_logits
    raise ValueError("Model forward must return (logits, embedding) or (logits, embedding, aux_logits)")
