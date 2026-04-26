from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

try:
    from tqdm import tqdm as _tqdm

    def tqdm(iterable, **kwargs):
        kwargs.setdefault("disable", not sys.stderr.isatty())
        return _tqdm(iterable, **kwargs)
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from ecg_acl.config import load_config
from ecg_acl.losses import InterAffinityContrastiveLoss, IntraMarginContrastiveLoss
from ecg_acl.metrics import AverageMeter, ConfusionMeter, ReliabilityZoneMeter, accuracy
from ecg_acl.model import build_ecg_model, unpack_model_output
from ecg_acl.train_utils import build_loaders, compute_class_weights, get_dataset_labels, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG classifier with affinity contrastive learning.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic ECG-like data for smoke testing.")
    parser.add_argument("--use-inter", action="store_true", help="Enable inter-class affinity contrastive loss.")
    parser.add_argument("--use-intra", action="store_true", help="Enable intra-class margin contrastive loss.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--work-dir", default=None, help="Override output directory.")
    parser.add_argument(
        "--selection-metric",
        choices=["acc", "macro_f1", "balanced_acc", "minority_recall", "minority_f1"],
        default=None,
        help="Validation metric used for best/top-k checkpoint selection.",
    )
    parser.add_argument("--save-top-k", type=int, default=None, help="Keep top-k epoch checkpoints by selection metric.")
    parser.add_argument(
        "--selection-metrics",
        default=None,
        help="Comma-separated validation metrics for independent top-k checkpoint saving.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Override gradient clipping max norm.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument(
        "--model",
        choices=["resnet1d", "fusion"],
        default=None,
        help="Override model architecture.",
    )
    parser.add_argument(
        "--branches",
        default=None,
        help="Comma-separated fusion branches, e.g. birnn,efficientnet,sequential,lenet.",
    )
    parser.add_argument("--aux-weight", type=float, default=None, help="Override fusion branch auxiliary loss weight.")
    parser.add_argument(
        "--loss-type",
        choices=["ce", "class_balanced_ce", "focal", "class_balanced_focal", "balanced_softmax"],
        default=None,
        help="Override classification loss for imbalanced data.",
    )
    parser.add_argument("--no-weighted-sampler", action="store_true", help="Disable class-balanced sampling.")
    parser.add_argument("--no-class-balanced-loss", action="store_true", help="Disable class-balanced CE weights.")
    return parser.parse_args()


def setup_logging(work_dir: Path, log_file: str | None) -> logging.Logger:
    work_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ecg_acl.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file:
        file_handler = logging.FileHandler(work_dir / log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def classification_loss(logits, labels, cfg, class_weights=None, class_counts=None):
    loss_type = str(cfg.get("imbalance", {}).get("loss_type", "ce"))
    if loss_type == "balanced_softmax":
        if class_counts is None:
            raise ValueError("class_counts are required for balanced_softmax")
        adjusted_logits = logits + class_counts.to(logits.device).clamp_min(1.0).log().view(1, -1)
        return F.cross_entropy(adjusted_logits, labels)

    weights = class_weights if "class_balanced" in loss_type else None
    ce = F.cross_entropy(logits, labels, weight=weights, reduction="none")
    if "focal" not in loss_type:
        return ce.mean()

    pt = torch.exp(-ce.detach())
    gamma = float(cfg.get("imbalance", {}).get("focal_gamma", 2.0))
    return (((1.0 - pt) ** gamma) * ce).mean()


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    inter_loss=None,
    intra_loss=None,
    cfg=None,
    epoch=0,
    class_weights=None,
    class_counts=None,
    grad_clip_norm=0.0,
):
    training = optimizer is not None
    model.train(training)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    ce_meter = AverageMeter()
    aux_meter = AverageMeter()
    inter_meter = AverageMeter()
    intra_meter = AverageMeter()
    grad_meter = AverageMeter()
    confusion = ConfusionMeter(int(cfg["data"]["num_classes"]))
    reliability = ReliabilityZoneMeter()

    iterator = tqdm(loader, leave=False, desc=f"{'train' if training else 'eval'} {epoch}")
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(training):
            logits, embeddings, aux_logits = unpack_model_output(model(x))
            ce = classification_loss(logits, y, cfg, class_weights, class_counts)
            aux = ce.new_zeros(())
            aux_weight = float(cfg.get("model", {}).get("aux_weight", 0.0))
            if aux_weight > 0 and aux_logits:
                aux_losses = [
                    classification_loss(branch_logits, y, cfg, class_weights, class_counts)
                    for branch_logits in aux_logits.values()
                ]
                aux = torch.stack(aux_losses).mean()
            inter = inter_loss(embeddings, y, epoch) if inter_loss is not None else ce.new_zeros(())
            intra = intra_loss(embeddings, y) if intra_loss is not None and training else ce.new_zeros(())
            affinity_cfg = cfg["affinity"]
            loss = (
                ce
                + aux_weight * aux
                + float(affinity_cfg["inter_weight"]) * inter
                + float(affinity_cfg["intra_weight"]) * intra
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = loss.new_zeros(())
                if grad_clip_norm and grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()
                if inter_loss is not None:
                    inter_loss.update(embeddings, y, logits)
            else:
                grad_norm = loss.new_zeros(())

        bs = x.size(0)
        loss_meter.update(float(loss.item()), bs)
        ce_meter.update(float(ce.item()), bs)
        aux_meter.update(float(aux.item()), bs)
        inter_meter.update(float(inter.item()), bs)
        intra_meter.update(float(intra.item()), bs)
        grad_meter.update(float(grad_norm.item()), bs)
        acc_meter.update(accuracy(logits.detach(), y), bs)
        confusion.update(logits.detach(), y)
        reliability.update(aux_logits, y)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=loss_meter.avg, acc=acc_meter.avg)

    metrics = {
        "loss": loss_meter.avg,
        "ce": ce_meter.avg,
        "aux": aux_meter.avg,
        "inter": inter_meter.avg,
        "intra": intra_meter.avg,
        "grad_norm": grad_meter.avg,
        "acc": acc_meter.avg,
    }
    metrics.update(confusion.compute())
    minority_classes = [int(idx) for idx in cfg.get("train", {}).get("minority_classes", [])]
    if minority_classes:
        recalls = metrics["per_class_recall"]
        f1_scores = metrics["per_class_f1"]
        valid = [idx for idx in minority_classes if 0 <= idx < len(recalls)]
        if valid:
            metrics["minority_recall"] = float(sum(recalls[idx] for idx in valid) / len(valid))
            metrics["minority_f1"] = float(sum(f1_scores[idx] for idx in valid) / len(valid))
    metrics.update(reliability.compute())
    return metrics


def _selection_score(metrics: dict, metric_name: str) -> float:
    if metric_name not in metrics:
        available = ", ".join(sorted(k for k, v in metrics.items() if isinstance(v, (int, float))))
        raise KeyError(f"selection_metric={metric_name!r} not found. Available scalar metrics: {available}")
    return float(metrics[metric_name])


def _metric_list(cfg: dict) -> list[str]:
    train_cfg = cfg.get("train", {})
    primary = str(train_cfg.get("selection_metric", "macro_f1"))
    raw = train_cfg.get("selection_metrics", [primary])
    if isinstance(raw, str):
        metrics = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        metrics = [str(item) for item in raw]
    if primary not in metrics:
        metrics.insert(0, primary)
    return list(dict.fromkeys(metrics))


def _clean_topk_files(work_dir: Path) -> None:
    for pattern in ("epoch_*.pt", "best_*.pt"):
        for path in work_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _build_scheduler(optimizer, cfg: dict):
    sched_cfg = cfg.get("scheduler", {})
    name = str(sched_cfg.get("name", "none")).lower()
    if name in {"", "none", "off", "disabled"}:
        return None, "none"
    if name == "cosine":
        return (
            CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(cfg["train"]["epochs"])),
                eta_min=float(sched_cfg.get("min_lr", 1e-5)),
            ),
            name,
        )
    if name == "plateau":
        return (
            ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(sched_cfg.get("factor", 0.5)),
                patience=int(sched_cfg.get("patience", 8)),
            ),
            name,
        )
    if name == "step":
        return (
            StepLR(
                optimizer,
                step_size=int(sched_cfg.get("step_size", 20)),
                gamma=float(sched_cfg.get("gamma", 0.5)),
            ),
            name,
        )
    raise ValueError(f"Unsupported scheduler.name: {name}")


def _step_scheduler(scheduler, scheduler_name: str, score: float) -> None:
    if scheduler is None:
        return
    if scheduler_name == "plateau":
        scheduler.step(score)
    else:
        scheduler.step()


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _save_checkpoint_candidates(
    model,
    cfg: dict,
    epoch: int,
    score: float,
    metric_name: str,
    work_dir: Path,
    candidates: list[dict],
    save_top_k: int,
) -> list[dict]:
    if save_top_k <= 0:
        return candidates

    safe_metric = metric_name.replace("/", "_")
    filename = f"epoch_{epoch:03d}_{safe_metric}_{score:.6f}.pt"
    path = work_dir / filename
    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "selection_metric": metric_name, "score": score}, path)

    candidates = [item for item in candidates if int(item["epoch"]) != epoch]
    candidates.append({"epoch": epoch, "score": score, "metric": metric_name, "path": filename})
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)

    kept = candidates[:save_top_k]
    dropped = candidates[save_top_k:]
    for item in dropped:
        drop_path = work_dir / str(item["path"])
        if drop_path.exists():
            drop_path.unlink()
    return kept


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.work_dir is not None:
        cfg["output"]["work_dir"] = args.work_dir
    if args.selection_metric is not None:
        cfg["train"]["selection_metric"] = args.selection_metric
    if args.selection_metrics is not None:
        cfg["train"]["selection_metrics"] = [item.strip() for item in args.selection_metrics.split(",") if item.strip()]
    if args.save_top_k is not None:
        cfg["train"]["save_top_k"] = args.save_top_k
    if args.grad_clip_norm is not None:
        cfg["train"]["grad_clip_norm"] = args.grad_clip_norm
    if args.early_stopping_patience is not None:
        cfg["train"]["early_stopping_patience"] = args.early_stopping_patience
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.branches is not None:
        cfg["model"]["branches"] = [name.strip() for name in args.branches.split(",") if name.strip()]
    if args.aux_weight is not None:
        cfg["model"]["aux_weight"] = args.aux_weight
    if args.no_weighted_sampler:
        cfg["imbalance"]["weighted_sampler"] = False
    if args.no_class_balanced_loss:
        cfg["imbalance"]["class_balanced_loss"] = False
    if args.loss_type is not None:
        cfg["imbalance"]["loss_type"] = args.loss_type

    work_dir = Path(cfg["output"]["work_dir"])
    logger = setup_logging(work_dir, cfg.get("output", {}).get("log_file", "train.log"))
    logger.info("work_dir=%s", work_dir)
    logger.info("config=%s", args.config or "<default>")

    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg["train"]["device"]))
    logger.info("device=%s", device)
    train_loader, val_loader, test_loader = build_loaders(cfg, args.synthetic, args.batch_size)

    model = build_ecg_model(cfg).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler, scheduler_name = _build_scheduler(optimizer, cfg)
    train_labels = get_dataset_labels(train_loader.dataset)
    class_counts = torch.bincount(
        torch.as_tensor(train_labels, dtype=torch.long),
        minlength=int(cfg["data"]["num_classes"]),
    ).float().to(device)
    logger.info("class_counts=%s", class_counts.detach().cpu().tolist())

    class_weights = None
    if bool(cfg.get("imbalance", {}).get("class_balanced_loss", False)) or "class_balanced" in str(cfg["imbalance"].get("loss_type", "")):
        class_weights = compute_class_weights(
            train_labels,
            num_classes=int(cfg["data"]["num_classes"]),
            beta=float(cfg["imbalance"]["effective_num_beta"]),
            max_weight=float(cfg["imbalance"]["max_class_weight"]),
        ).to(device)
        logger.info("class_weights=%s", class_weights.detach().cpu().tolist())

    inter_loss = None
    if args.use_inter:
        inter_loss = InterAffinityContrastiveLoss(
            num_classes=int(cfg["data"]["num_classes"]),
            embedding_dim=int(cfg["model"]["embedding_dim"]),
            top_k=int(cfg["affinity"]["top_k"]),
            start_epoch=int(cfg["affinity"]["start_epoch"]),
            overlap_threshold=int(cfg["affinity"]["overlap_threshold"]),
            temperature=float(cfg["affinity"]["temperature"]),
        ).to(device)
    intra_loss = None
    if args.use_intra:
        intra_loss = IntraMarginContrastiveLoss(
            memory_size=int(cfg["affinity"]["memory_size"]),
            temperature=float(cfg["affinity"]["temperature"]),
            margin=float(cfg["affinity"]["margin"]),
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    selection_metric = str(cfg["train"].get("selection_metric", "macro_f1"))
    selection_metrics = _metric_list(cfg)
    save_top_k = int(cfg["train"].get("save_top_k", 1))
    if bool(cfg["train"].get("clean_topk", True)):
        _clean_topk_files(work_dir)
    best_scores = {metric: -1.0 for metric in selection_metrics}
    best_checkpoints: dict[str, dict] = {}
    checkpoint_candidates: dict[str, list[dict]] = {metric: [] for metric in selection_metrics}
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 0.0) or 0.0)
    early_patience = int(cfg["train"].get("early_stopping_patience", 0) or 0)
    early_min_delta = float(cfg["train"].get("early_stopping_min_delta", 0.0) or 0.0)
    stale_epochs = 0
    history = []
    logger.info(
        "selection_metric=%s selection_metrics=%s save_top_k=%d scheduler=%s grad_clip_norm=%.4g early_stopping_patience=%d",
        selection_metric,
        selection_metrics,
        save_top_k,
        scheduler_name,
        grad_clip_norm,
        early_patience,
    )

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        if inter_loss is not None and epoch >= int(cfg["affinity"]["start_epoch"]):
            inter_loss.rebuild_affinity_sets()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            inter_loss,
            intra_loss,
            cfg,
            epoch,
            class_weights,
            class_counts,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = run_epoch(model, val_loader, device, None, inter_loss, None, cfg, epoch, class_weights, class_counts)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        primary_score = _selection_score(val_metrics, selection_metric)
        logger.info(
            "epoch=%03d lr=%.6g train_loss=%.4f train_acc=%.4f train_grad=%.4f "
            "train_aux=%.4f train_macro_f1=%.4f val_loss=%.4f val_acc=%.4f "
            "val_aux=%.4f val_macro_f1=%.4f val_bal_acc=%.4f val_min_recall=%.4f",
            epoch,
            _current_lr(optimizer),
            train_metrics["loss"],
            train_metrics["acc"],
            train_metrics["grad_norm"],
            train_metrics["aux"],
            train_metrics["macro_f1"],
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["aux"],
            val_metrics["macro_f1"],
            val_metrics["balanced_acc"],
            val_metrics["minority_recall"],
        )

        primary_improved = False
        for metric_name in selection_metrics:
            score = _selection_score(val_metrics, metric_name)
            checkpoint_candidates[metric_name] = _save_checkpoint_candidates(
                model,
                cfg,
                epoch,
                score,
                metric_name,
                work_dir,
                checkpoint_candidates[metric_name],
                save_top_k,
            )
            if score > best_scores[metric_name] + early_min_delta:
                best_scores[metric_name] = score
                safe_metric = metric_name.replace("/", "_")
                checkpoint = {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "selection_metric": metric_name,
                    "score": score,
                }
                torch.save(checkpoint, work_dir / f"best_{safe_metric}.pt")
                best_checkpoints[metric_name] = {
                    "epoch": epoch,
                    "score": score,
                    "path": f"best_{safe_metric}.pt",
                }
                if metric_name == selection_metric:
                    torch.save(checkpoint, work_dir / "best.pt")
                    primary_improved = True
        if primary_improved:
            stale_epochs = 0
        else:
            stale_epochs += 1

        _step_scheduler(scheduler, scheduler_name, primary_score)
        if early_patience > 0 and stale_epochs >= early_patience:
            logger.info(
                "early_stop epoch=%03d stale_epochs=%d best_%s=%.6f",
                epoch,
                stale_epochs,
                selection_metric,
                best_scores[selection_metric],
            )
            break

    checkpoint = torch.load(work_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(
        model, test_loader, device, None, inter_loss, None, cfg, checkpoint["epoch"], class_weights, class_counts
    )
    save_json(
        {
            "history": history,
            "best_epoch": checkpoint["epoch"],
            "selection_metric": selection_metric,
            "selection_metrics": selection_metrics,
            "best_score": best_scores[selection_metric],
            "best_scores": best_scores,
            "best_checkpoints": best_checkpoints,
            "checkpoint_candidates": checkpoint_candidates,
            "test": test_metrics,
        },
        work_dir / "metrics.json",
    )
    logger.info(
        "best_epoch=%s test_loss=%.4f test_acc=%.4f test_macro_f1=%.4f test_bal_acc=%.4f",
        checkpoint["epoch"],
        test_metrics["loss"],
        test_metrics["acc"],
        test_metrics["macro_f1"],
        test_metrics["balanced_acc"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("ecg_acl.train").exception("training failed")
        raise
