"""
calibration.py
==============
Are the classifiers' confidence scores trustworthy? When a model says 90%,
it should be right about 90% of the time.

- ``expected_calibration_error``: ECE with 15 equal-width bins, computed by
  torchmetrics' MulticlassCalibrationError (top-label, L1).
- ``fit_temperature``: temperature scaling (Guo et al., "On Calibration of
  Modern Neural Networks", ICML 2017), following the reference
  implementation by G. Pleiss (github.com/gpleiss/temperature_scaling): one
  scalar T > 0 divides the logits and is fitted by minimizing NLL on the
  validation split with L-BFGS. Dividing by T never changes the predicted
  class, so accuracy is unchanged.
- ``reliability_curve``: per-bin confidence vs. accuracy for diagrams.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_BINS = 15


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = N_BINS) -> float:
    from torchmetrics.classification import MulticlassCalibrationError
    metric = MulticlassCalibrationError(num_classes=probs.shape[1], n_bins=n_bins, norm="l1")
    return float(metric(probs, labels))


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature minimizing the NLL of ``logits / T`` (validation data only)."""
    log_t = nn.Parameter(torch.zeros(1))          # T = exp(log_t) stays positive
    # Strong-Wolfe line search: with a small fixed step (lr=0.01) L-BFGS stopped
    # early, e.g. at T = 2.43 where the NLL optimum is 2.97.
    optimizer = torch.optim.LBFGS([log_t], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp())


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    """(bin centres, accuracy per bin, confidence per bin, count per bin)."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    acc = np.array([(pred[idx == b] == labels[idx == b]).mean() if (idx == b).any() else np.nan
                    for b in range(n_bins)])
    mean_conf = np.array([conf[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)])
    counts = np.bincount(idx, minlength=n_bins)
    return (edges[:-1] + edges[1:]) / 2, acc, mean_conf, counts
