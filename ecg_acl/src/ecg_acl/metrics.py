from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return preds.eq(labels).float().mean().item()


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class ConfusionMeter:
    def __init__(self, num_classes: int):
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        preds = logits.argmax(dim=1).detach().cpu()
        labels = labels.detach().cpu()
        for target, pred in zip(labels, preds):
            self.matrix[int(target), int(pred)] += 1

    def compute(self) -> dict[str, float | list[float] | list[list[int]]]:
        matrix = self.matrix.float()
        tp = matrix.diag()
        support = matrix.sum(dim=1).clamp_min(1.0)
        pred_count = matrix.sum(dim=0).clamp_min(1.0)
        recall = tp / support
        precision = tp / pred_count
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        minority_slice = slice(1, None) if len(recall) > 1 else slice(0, None)
        total = matrix.sum().clamp_min(1.0)
        return {
            "balanced_acc": float(recall.mean().item()),
            "macro_f1": float(f1.mean().item()),
            "minority_recall": float(recall[minority_slice].mean().item()),
            "minority_f1": float(f1[minority_slice].mean().item()),
            "per_class_precision": [float(x) for x in precision.tolist()],
            "per_class_recall": [float(x) for x in recall.tolist()],
            "per_class_f1": [float(x) for x in f1.tolist()],
            "confusion": self.matrix.tolist(),
            "samples": int(total.item()),
        }


class ReliabilityZoneMeter:
    def __init__(self):
        self.correct = 0
        self.misclassified = 0
        self.disagreement = 0
        self.total = 0

    @torch.no_grad()
    def update(self, aux_logits: dict[str, torch.Tensor], labels: torch.Tensor) -> None:
        if not aux_logits:
            return
        preds = torch.stack([logits.argmax(dim=1) for logits in aux_logits.values()], dim=1).detach().cpu()
        labels = labels.detach().cpu()
        unanimous = preds.eq(preds[:, :1]).all(dim=1)
        agreed_pred = preds[:, 0]
        self.correct += int((unanimous & agreed_pred.eq(labels)).sum().item())
        self.misclassified += int((unanimous & ~agreed_pred.eq(labels)).sum().item())
        self.disagreement += int((~unanimous).sum().item())
        self.total += int(labels.numel())

    def compute(self) -> dict[str, float]:
        if self.total == 0:
            return {}
        return {
            "reliability_correct_zone": self.correct / self.total,
            "reliability_misclassification_zone": self.misclassified / self.total,
            "reliability_disagreement_zone": self.disagreement / self.total,
        }
