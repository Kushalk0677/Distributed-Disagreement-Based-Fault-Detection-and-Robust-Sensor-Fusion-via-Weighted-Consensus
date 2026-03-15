"""
fusion.py — all fusion methods

Centralized baselines
  1. fuse_average        — unweighted mean
  2. fuse_trimmed_mean   — Byzantine-resilient (drop top/bottom k%)

Distributed baselines
  3. fuse_local_median   — each sensor uses median of neighbors; no consensus
  4. fuse_consensus_plain — iterative averaging (equal weights)

Proposed
  5. fuse_proposed       — enhanced disagreement weights + weighted consensus

Supporting functions
  - compute_disagreement, smooth_disagreement, compute_trend
  - compute_enhanced_disagreement, compute_weights
  - build_reliability_weighted_consensus
  - adaptive_fault_threshold
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Centralized baselines
# ─────────────────────────────────────────────────────────────────────────────

def fuse_average(Y: np.ndarray) -> np.ndarray:
    return Y.mean(axis=0)


def fuse_trimmed_mean(Y: np.ndarray, trim: float = 0.10) -> np.ndarray:
    """
    Remove top and bottom `trim` fraction of sensors at each time step.
    Byzantine-resilient centralized baseline.
    """
    N    = Y.shape[0]
    k    = max(1, int(N * trim))
    Ys   = np.sort(Y, axis=0)
    return Ys[k : N - k].mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed baseline — local median
# ─────────────────────────────────────────────────────────────────────────────

def fuse_local_median(Y: np.ndarray, neighbors: list) -> np.ndarray:
    """
    Each sensor estimates the signal as the median of its neighbors'
    measurements. Global estimate = average of all local estimates.

    This is a *truly distributed* method: each sensor only needs its
    one-hop neighborhood. No consensus rounds needed.

    Weakness: if >50% of a sensor's neighbors are faulty (clustered faults
    or high-noise), the local median is pulled toward the faulty value.
    """
    N, T    = Y.shape
    X_local = np.zeros((N, T))
    for i in range(N):
        nbrs = neighbors[i]
        if len(nbrs) > 0:
            X_local[i] = np.median(Y[nbrs, :], axis=0)
        else:
            X_local[i] = Y[i]
    return X_local.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Consensus engine
# ─────────────────────────────────────────────────────────────────────────────

def fuse_consensus_plain(Y: np.ndarray, P: np.ndarray,
                          n_iter: int = 40) -> np.ndarray:
    """
    Plain iterative consensus with Metropolis-Hastings weights.
    Converges to the unweighted average (equivalent to fuse_average in limit).
    Serves as the distributed-but-unweighted baseline.

    Parameters
    ----------
    Y      : (N, T) measurements
    P      : (N, N) Metropolis mixing matrix from network.build_metropolis_weights
    n_iter : consensus iterations (≥ graph diameter for convergence)
    """
    X = Y.astype(float)
    for _ in range(n_iter):
        X = P @ X
    return X.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Proposed: disagreement weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_disagreement(Y: np.ndarray, neighbors: list) -> np.ndarray:
    """d_i(t) = | y_i(t) - median(y_j(t), j ∈ N_i) |  → (N, T)"""
    N, T = Y.shape
    D    = np.zeros((N, T))
    for i in range(N):
        nbrs = neighbors[i]
        if len(nbrs) > 0:
            D[i] = np.abs(Y[i] - np.median(Y[nbrs], axis=0))
    return D


def smooth_disagreement(D: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """D_smooth(t) = α D_smooth(t-1) + (1-α) D(t)  → (N, T)"""
    N, T  = D.shape
    Ds    = np.zeros_like(D)
    Ds[:, 0] = D[:, 0]
    for t in range(1, T):
        Ds[:, t] = alpha * Ds[:, t - 1] + (1 - alpha) * D[:, t]
    return Ds


def compute_trend(Ds: np.ndarray, lag: int = 10) -> np.ndarray:
    """
    trend_i(t) = max(0, D_smooth_i(t) - D_smooth_i(t - lag))

    Captures monotonically growing bias (drift faults) that appear small
    at any single instant but grow consistently over time.
    """
    trend          = np.zeros_like(Ds)
    trend[:, lag:] = np.maximum(0.0, Ds[:, lag:] - Ds[:, :-lag])
    return trend


def compute_enhanced_disagreement(Ds: np.ndarray,
                                   beta: float = 2.0,
                                   lag: int = 10) -> np.ndarray:
    """
    D_enhanced = D_smooth + β · trend

    Combines:
      • Magnitude term  → detects stuck / malicious faults quickly
      • Trend term      → detects slow drift faults over time
    """
    return Ds + beta * compute_trend(Ds, lag=lag)


def compute_weights(D_enh: np.ndarray) -> np.ndarray:
    """w_i(t) = 1 / (1 + D_enhanced_i(t))  ∈ (0, 1]"""
    return 1.0 / (1.0 + D_enh)


# ─────────────────────────────────────────────────────────────────────────────
# Proposed: reliability-weighted consensus
# ─────────────────────────────────────────────────────────────────────────────

def fuse_proposed(Y: np.ndarray, neighbors: list,
                  P: np.ndarray,
                  alpha: float = 0.85,
                  beta: float  = 2.0,
                  lag: int     = 10,
                  n_iter: int  = 40) -> tuple:
    """
    Full proposed pipeline:

    1. Compute enhanced disagreement (magnitude + trend)
    2. Convert to reliability weights
    3. Run reliability-weighted iterative consensus

    Why this beats local median under heterogeneous noise
    ─────────────────────────────────────────────────────
    High-noise sensors have persistently high disagreement with low-noise
    neighbors → they receive persistently low reliability weights → the
    weighted consensus automatically up-weights low-noise sensors.
    Local median cannot distinguish noise quality; this method does.

    Why this beats local median under clustered faults
    ──────────────────────────────────────────────────
    Even if a sensor's immediate neighbors are all faulty (so local
    median = faulty value), the weighted consensus propagates reliable
    information from distant healthy sensors over K iterations.

    Parameters
    ----------
    Y         : (N, T)
    neighbors : list of lists
    P         : (N, N) Metropolis mixing matrix
    alpha     : temporal smoothing coefficient
    beta      : trend-term weight
    lag       : trend lookback window
    n_iter    : consensus iterations

    Returns
    -------
    s_hat      : (T,)  global estimate
    W          : (N, T) reliability weights
    D_smooth   : (N, T)
    D_enhanced : (N, T)
    """
    D_raw  = compute_disagreement(Y, neighbors)
    Ds     = smooth_disagreement(D_raw, alpha=alpha)
    D_enh  = compute_enhanced_disagreement(Ds, beta=beta, lag=lag)
    W      = compute_weights(D_enh)                        # (N, T)

    # Reliability-weighted consensus
    # At each iteration: x_i ← Σ_j P_ij · w_j · x_j / Σ_j P_ij · w_j
    X = Y.astype(float).copy()
    for _ in range(n_iter):
        X_new = np.zeros_like(X)
        for i in range(len(neighbors)):
            nbrs_i = neighbors[i] + [i]
            mix_w  = P[i, nbrs_i][:, np.newaxis]          # (|Ni|+1, 1)
            rel_w  = W[nbrs_i, :]                          # (|Ni|+1, T)
            total  = mix_w * rel_w                         # (|Ni|+1, T)
            denom  = total.sum(axis=0) + 1e-12
            X_new[i] = (total * X[nbrs_i]).sum(axis=0) / denom
        X = X_new

    s_hat = X.mean(axis=0)
    return s_hat, W, Ds, D_enh


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive fault-detection threshold
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_fault_threshold(W: np.ndarray,
                              time_start: int = 0,
                              k: float = 1.5) -> float:
    """
    threshold = mean(w̄) − k · std(w̄)

    where w̄_i = time-averaged weight of sensor i after fault onset.
    Sensors below threshold are classified as faulty.
    """
    w_bar = W[:, time_start:].mean(axis=1)
    return float(w_bar.mean() - k * w_bar.std())
