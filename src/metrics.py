"""
metrics.py — evaluation metrics

Estimation accuracy:  MSE, RMSE
Detection quality:    FDR, FAR, Precision, Recall, F1
Communication cost:   rounds_per_timestep
"""

import numpy as np


def mse(s_true, s_hat):
    return float(np.mean((s_true - s_hat) ** 2))


def rmse(s_true, s_hat):
    return float(np.sqrt(mse(s_true, s_hat)))


def detection_metrics(fault_mask: np.ndarray,
                      predicted_mask: np.ndarray) -> dict:
    """Binary classification metrics for fault detection."""
    TP = int(np.sum( predicted_mask &  fault_mask))
    FP = int(np.sum( predicted_mask & ~fault_mask))
    FN = int(np.sum(~predicted_mask &  fault_mask))
    TN = int(np.sum(~predicted_mask & ~fault_mask))

    FDR  = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # recall
    FAR  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    F1   = 2 * prec * FDR / (prec + FDR) if (prec + FDR) > 0 else 0.0

    return dict(FDR=FDR, FAR=FAR, precision=prec, recall=FDR, F1=F1,
                TP=TP, FP=FP, FN=FN, TN=TN)


def communication_overhead(n_iter_consensus: int) -> dict:
    """
    Rounds of neighbor communication per time step.
    Local median    : 1 round  (broadcast measurement)
    Proposed        : 1 + n_iter rounds (1 for disagreement + consensus)
    Plain consensus : 1 + n_iter rounds
    """
    return dict(
        average         = 0,           # centralized, no comm modeled
        trimmed_mean    = 0,
        local_median    = 1,
        plain_consensus = 1 + n_iter_consensus,
        proposed        = 1 + n_iter_consensus,
        dist_kf         = n_iter_consensus,  # consensus only (no separate disagr. round)
    )


def ci95(values: list) -> tuple:
    """95% confidence interval from a list of scalar values."""
    arr  = np.array(values)
    mean = arr.mean()
    sem  = arr.std(ddof=1) / np.sqrt(len(arr))
    return mean, 1.96 * sem
