"""
detection.py — fault detection algorithms

1. DisagreementDetector  — proposed method (enhanced disagreement + adaptive threshold)
2. CUSUMDetector         — CUSUM on per-sensor innovations (standard baseline)
3. EWMADetector          — EWMA control chart (additional baseline)

All detectors share a common interface:
    detector.fit_predict(Y, neighbors, fault_onset) → (predicted_mask, scores)
"""

import numpy as np
from src.fusion import (compute_disagreement, smooth_disagreement,
                         compute_enhanced_disagreement, compute_weights,
                         adaptive_fault_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Proposed
# ─────────────────────────────────────────────────────────────────────────────

class DisagreementDetector:
    """
    Fault detection based on enhanced disagreement scores.
    Uses adaptive threshold (data-driven, no prior fault knowledge needed).
    """

    def __init__(self, alpha: float = 0.85, beta: float = 2.0,
                 lag: int = 10, k: float = 1.5):
        self.alpha = alpha
        self.beta  = beta
        self.lag   = lag
        self.k     = k

    def fit_predict(self, Y: np.ndarray, neighbors: list,
                    fault_onset: int = 0):
        D_raw  = compute_disagreement(Y, neighbors)
        Ds     = smooth_disagreement(D_raw, alpha=self.alpha)
        D_enh  = compute_enhanced_disagreement(Ds, beta=self.beta, lag=self.lag)
        W      = compute_weights(D_enh)
        thresh = adaptive_fault_threshold(W, time_start=fault_onset, k=self.k)
        w_bar  = W[:, fault_onset:].mean(axis=1)
        pred   = w_bar < thresh
        return pred, W, D_enh


# ─────────────────────────────────────────────────────────────────────────────
# CUSUM
# ─────────────────────────────────────────────────────────────────────────────

class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) change detection on per-sensor innovation.

    Innovation: i_i(t) = | y_i(t) − median(y_j(t), j∈N_i) |
    CUSUM:      S_i(t) = max(0, S_i(t-1) + i_i(t) − slack)
    Detection:  S_i(T_eval:) mean > threshold

    Parameters
    ----------
    slack     : allowance per step (set ≈ expected noise level)
    threshold : cumulative sum threshold for declaration
    """

    def __init__(self, slack: float = 0.08, threshold: float = 1.5):
        self.slack     = slack
        self.threshold = threshold

    def fit_predict(self, Y: np.ndarray, neighbors: list,
                    fault_onset: int = 0):
        N, T = Y.shape
        innov = compute_disagreement(Y, neighbors)       # (N, T)

        S = np.zeros((N, T))
        for t in range(1, T):
            S[:, t] = np.maximum(0.0,
                                  S[:, t - 1] + innov[:, t] - self.slack)

        S_mean = S[:, fault_onset:].mean(axis=1)
        pred   = S_mean > self.threshold
        return pred, S_mean, innov


# ─────────────────────────────────────────────────────────────────────────────
# EWMA
# ─────────────────────────────────────────────────────────────────────────────

class EWMADetector:
    """
    EWMA (Exponentially Weighted Moving Average) control chart.

    EWMA_i(t) = λ · i_i(t) + (1-λ) · EWMA_i(t-1)
    UCL       = μ_0 + L · σ_0 · sqrt(λ / (2-λ))

    Parameters estimated from the pre-fault window.
    """

    def __init__(self, lam: float = 0.2, L: float = 3.0):
        self.lam = lam
        self.L   = L

    def fit_predict(self, Y: np.ndarray, neighbors: list,
                    fault_onset: int = 50):
        N, T  = Y.shape
        innov = compute_disagreement(Y, neighbors)       # (N, T)

        # Estimate baseline stats from pre-fault window
        pre   = innov[:, :fault_onset]
        mu0   = pre.mean(axis=1, keepdims=True)
        sig0  = pre.std(axis=1, keepdims=True) + 1e-8

        # Normalize innovations
        z = (innov - mu0) / sig0

        # EWMA
        E = np.zeros((N, T))
        E[:, 0] = z[:, 0]
        for t in range(1, T):
            E[:, t] = self.lam * z[:, t] + (1 - self.lam) * E[:, t - 1]

        ucl  = self.L * np.sqrt(self.lam / (2 - self.lam))
        pred = E[:, fault_onset:].mean(axis=1) > ucl
        return pred, E, ucl
