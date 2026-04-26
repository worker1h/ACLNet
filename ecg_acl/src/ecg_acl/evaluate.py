from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        matrix[int(target), int(pred)] += 1
    return matrix


def per_class_accuracy(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=1).clip(min=1)
    return np.diag(matrix) / denom
