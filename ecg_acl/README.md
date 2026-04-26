# ECG Affinity Contrastive Learning

This folder contains an ECG classification experiment adapted from ACLNet's
affinity contrastive learning idea.

## Layout

- `configs/mitbih.yaml`: default MIT-BIH beat classification configuration.
- `scripts/download_mitbih.py`: optional MIT-BIH downloader using `wfdb`.
- `scripts/preprocess_mitbih.py`: converts MIT-BIH records into train/val/test `.npz` files.
- `src/ecg_acl`: reusable dataset, model, loss, and training utilities.
- `train.py`: training entry point.

## Data

The training code expects preprocessed files:

```text
ecg_acl/data/mitbih/processed/train.npz
ecg_acl/data/mitbih/processed/val.npz
ecg_acl/data/mitbih/processed/test.npz
```

Each `.npz` file contains:

- `x`: float32 ECG beats, shape `[N, C, T]`
- `y`: int64 class ids, shape `[N]`

To download and preprocess MIT-BIH:

```shell
pip install wfdb
python ecg_acl/scripts/download_mitbih.py --out ecg_acl/data/mitbih/raw
python ecg_acl/scripts/preprocess_mitbih.py --raw ecg_acl/data/mitbih/raw --out ecg_acl/data/mitbih/processed
```

To build the RR + neighbor-beat variant:

```shell
python ecg_acl/scripts/preprocess_mitbih_context_rr.py --raw ecg_acl/data/mitbih/raw --out ecg_acl/data/mitbih_context_rr_record/processed --split-mode record
```

This writes 7-channel beats by default: previous/current/next beat waveforms
plus previous RR ratio, next RR ratio, RR delta ratio, and local RR ratio.
The default split mode holds out whole DS1 records for validation to reduce
patient/record leakage.

## Train

Baseline:

```shell
python ecg_acl/train.py --config ecg_acl/configs/mitbih.yaml
```

BiRNN + PHM branch fusion:

```shell
python ecg_acl/train.py --config ecg_acl/configs/mitbih_fusion.yaml
```

Affinity contrastive training:

```shell
python ecg_acl/train.py --config ecg_acl/configs/mitbih.yaml --use-inter --use-intra
```

BiRNN + PHM fusion with ACL losses:

```shell
python ecg_acl/train.py --config ecg_acl/configs/mitbih_fusion.yaml --use-inter --use-intra
```

BiRNN + PHM fusion with RR and neighbor-beat context:

```shell
python ecg_acl/train.py --config ecg_acl/configs/mitbih_fusion_context_rr.yaml --use-inter --use-intra
```

MIT-BIH is highly imbalanced. The default config enables weighted sampling,
class-balanced CE, and checkpoint selection by `macro_f1`. Optional
`focal` / `class_balanced_focal`, `balanced_softmax`, and minority-only ECG
augmentation are available in the config for ablations. To reproduce the plain
imbalanced setting, disable the balancing options explicitly:

```shell
python ecg_acl/train.py --use-inter --use-intra --no-weighted-sampler --no-class-balanced-loss
```

Useful fusion ablations:

```shell
# BiRNN-only branch with ACL-compatible embedding.
python ecg_acl/train.py --config ecg_acl/configs/mitbih_fusion.yaml --branches birnn --aux-weight 0.2

# PHM CNN branches without BiRNN.
python ecg_acl/train.py --config ecg_acl/configs/mitbih_fusion.yaml --branches efficientnet,sequential,lenet --aux-weight 0.2

# Disable auxiliary branch supervision.
python ecg_acl/train.py --config ecg_acl/configs/mitbih_fusion.yaml --aux-weight 0.0

# Use focal loss for imbalanced classes.
python ecg_acl/train.py --config ecg_acl/configs/mitbih_fusion.yaml --loss-type class_balanced_focal
```

Checkpoint selection can be shifted away from plain `macro_f1`:

```shell
# Keep top-5 epochs by validation balanced accuracy.
python ecg_acl/train.py --config ecg_acl/configs/mitbih_fusion_select_balanced.yaml --use-inter --use-intra

# Keep top-5 epochs by non-normal class recall.
python ecg_acl/train.py --config ecg_acl/configs/mitbih_fusion_select_minority_recall.yaml --use-inter --use-intra
```

Configs can also keep top-k candidates for multiple validation metrics:

```yaml
train:
  selection_metric: balanced_acc
  selection_metrics: [balanced_acc, macro_f1, minority_f1]
  save_top_k: 5
```

Training writes structured logs to `train.log`, supports cosine/step/plateau
learning-rate scheduling, gradient clipping, and optional early stopping.

Post-processing can be evaluated without retraining:

```shell
# Tune class thresholds on validation data, then evaluate on test data.
python ecg_acl/evaluate.py --checkpoint ecg_acl/work_dirs/fusion_acl_full/best.pt --decision-rule threshold --tune-thresholds --target-metric balanced_acc

# Reject low-confidence samples into an extra review column.
python ecg_acl/evaluate.py --checkpoint ecg_acl/work_dirs/fusion_acl_full/best.pt --decision-rule reject --reject-threshold 0.75
```

Experiment metrics can be visualized after training:

```shell
python ecg_acl/scripts/visualize_experiment_results.py --metrics ecg_acl/work_dirs/fusion_context_rr_record_acl/metrics.json
```

The script writes training curves, final metric bars, per-class metric bars,
a normalized confusion matrix, top-k checkpoint CSV, and `summary.md` under
`<work_dir>/visualizations/`.

`metrics.json` records PHM-style reliability zones when fusion branches are
enabled:

- `reliability_correct_zone`: all branches agree and are correct.
- `reliability_misclassification_zone`: all branches agree on a wrong class.
- `reliability_disagreement_zone`: branches disagree.

Smoke test without real data:

```shell
python ecg_acl/train.py --synthetic --epochs 1 --batch-size 16
```
