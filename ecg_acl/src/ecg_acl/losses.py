from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class InterAffinityContrastiveLoss(nn.Module):
    """Class-prototype contrastive loss using confusion-derived affinity sets."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        top_k: int = 5,
        start_epoch: int = 20,
        overlap_threshold: int = 2,
        momentum: float = 0.9,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.top_k = min(top_k, max(1, num_classes - 1))
        self.start_epoch = start_epoch
        self.overlap_threshold = overlap_threshold
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer("prototypes", F.normalize(torch.randn(num_classes, embedding_dim), dim=1))
        self.register_buffer("prototype_seen", torch.zeros(num_classes))
        self.register_buffer("confusion", torch.zeros(num_classes, num_classes))
        self.affinity_sets: list[list[int]] = [[] for _ in range(num_classes)]

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor) -> None:
        embeddings = F.normalize(embeddings.detach(), dim=1)
        labels = labels.detach()
        preds = logits.detach().argmax(dim=1)

        for cls in labels.unique():
            cls_idx = int(cls.item())
            mask = labels == cls
            cls_proto = embeddings[mask].mean(dim=0)
            cls_proto = F.normalize(cls_proto, dim=0)
            if self.prototype_seen[cls_idx] == 0:
                self.prototypes[cls_idx] = cls_proto
            else:
                updated = self.momentum * self.prototypes[cls_idx] + (1.0 - self.momentum) * cls_proto
                self.prototypes[cls_idx] = F.normalize(updated, dim=0)
            self.prototype_seen[cls_idx] += mask.sum()

        wrong = preds != labels
        if wrong.any():
            flat = labels[wrong] * self.num_classes + preds[wrong]
            counts = torch.bincount(flat, minlength=self.num_classes * self.num_classes)
            self.confusion += counts.view(self.num_classes, self.num_classes).to(self.confusion)

    @torch.no_grad()
    def rebuild_affinity_sets(self) -> None:
        pairwise = torch.zeros_like(self.confusion, dtype=torch.bool)
        for cls in range(self.num_classes):
            scores = self.confusion[cls].clone()
            scores[cls] = -1
            if scores.max() <= 0:
                continue
            k = min(self.top_k, int((scores > 0).sum().item()))
            if k <= 0:
                continue
            indices = torch.topk(scores, k=k).indices.tolist()
            pairwise[cls, indices] = True

        sets: list[list[int]] = []
        for cls in range(self.num_classes):
            candidates = set(torch.where(pairwise[cls])[0].tolist())
            cls_neighbors = pairwise[cls]
            for other in range(self.num_classes):
                if other == cls:
                    continue
                overlap = torch.logical_and(cls_neighbors, pairwise[other]).sum().item()
                if overlap >= self.overlap_threshold:
                    candidates.add(other)
            candidates.discard(cls)
            sets.append(sorted(candidates))
        self.affinity_sets = sets

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, epoch: int) -> torch.Tensor:
        if epoch < self.start_epoch:
            return embeddings.new_zeros(())
        embeddings = F.normalize(embeddings, dim=1)
        prototypes = F.normalize(self.prototypes.detach(), dim=1)
        loss_sum = embeddings.new_zeros(())
        count = 0
        for label in labels.unique():
            cls = int(label.item())
            candidates = [cls] + self.affinity_sets[cls]
            if len(candidates) == 1:
                candidates = list(range(self.num_classes))
            local_proto = prototypes[candidates]
            mask = labels == cls
            local_logits = torch.matmul(embeddings[mask], local_proto.T) / self._adaptive_temperature(len(candidates))
            targets = torch.zeros(local_logits.size(0), dtype=torch.long, device=embeddings.device)
            loss_sum = loss_sum + F.cross_entropy(local_logits, targets, reduction="sum")
            count += int(mask.sum().item())
        if count == 0:
            return embeddings.new_zeros(())
        return loss_sum / count

    def _adaptive_temperature(self, family_size: int) -> float:
        if family_size <= 6:
            return self.temperature
        if family_size <= 12:
            return self.temperature * 2.0
        return self.temperature * 4.0


class IntraMarginContrastiveLoss(nn.Module):
    """Memory-bank supervised contrastive loss with a positive margin."""

    def __init__(self, memory_size: int = 1024, temperature: float = 0.1, margin: float = 0.1):
        super().__init__()
        self.memory_size = memory_size
        self.temperature = temperature
        self.margin = margin
        self._features: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        embeddings = F.normalize(embeddings, dim=1)
        batch_size = embeddings.size(0)

        with torch.no_grad():
            for feature, label in zip(embeddings.detach().cpu(), labels.detach().cpu()):
                self._features.append(feature)
                self._labels.append(label)
            if len(self._features) > self.memory_size:
                overflow = len(self._features) - self.memory_size
                del self._features[:overflow]
                del self._labels[:overflow]

        if len(self._features) < max(batch_size * 2, 16):
            return embeddings.new_zeros(())

        memory_features = torch.stack(self._features).to(device)
        memory_labels = torch.stack(self._labels).to(device).view(-1)
        logits = torch.matmul(embeddings, memory_features.T) / self.temperature
        same = labels.view(-1, 1).eq(memory_labels.view(1, -1))
        positive_logits = logits - self.margin
        exp_pos = torch.exp(positive_logits) * same.float()
        exp_neg = torch.exp(logits) * (~same).float()
        denominator = exp_pos.sum(dim=1) + exp_neg.sum(dim=1) + 1e-8
        numerator = exp_pos.sum(dim=1) + 1e-8
        valid = same.any(dim=1)
        if not valid.any():
            return embeddings.new_zeros(())
        return -torch.log(numerator[valid] / denominator[valid]).mean()
