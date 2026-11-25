"""Evaluation metrics. The competition metric is multi-class log loss."""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of the true class (probabilities are clipped and renormalised)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    y = np.asarray(y_true, dtype=int)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float((np.asarray(probs).argmax(axis=1) == np.asarray(y_true)).mean())


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE: gap between confidence and accuracy, averaged over confidence bins."""
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def summarize(y_true: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "log_loss": log_loss(y_true, probs),
        "accuracy": accuracy(y_true, probs),
        "ece": expected_calibration_error(y_true, probs),
        "n": int(len(y_true)),
    }
