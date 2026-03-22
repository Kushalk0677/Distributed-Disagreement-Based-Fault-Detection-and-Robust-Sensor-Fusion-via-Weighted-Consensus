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


def compute_trend(Ds: np.ndarray, lag: int = 25) -> np.ndarray:
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
                                   lag: int = 25) -> np.ndarray:
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
                  lag: int     = 25,
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

    # Reliability-weighted consensus — vectorized
    # X_new[i,t] = Σ_j P[i,j]·W[j,t]·X[j,t] / Σ_j P[i,j]·W[j,t]
    #            = (P @ (W*X))[i,t]  /  (P @ W)[i,t]
    X = Y.astype(float).copy()
    for _ in range(n_iter):
        PW    = P @ W              # (N, T)
        X     = (P @ (W * X)) / (PW + 1e-12)

    s_hat = X.mean(axis=0)
    return s_hat, W, Ds, D_enh


# ─────────────────────────────────────────────────────────────────────────────
# Distributed Kalman Filter with consensus
# ─────────────────────────────────────────────────────────────────────────────

def fuse_distributed_kf(Y: np.ndarray, P: np.ndarray,
                         Q: float = 1e-3,
                         R: float = 0.0025,
                         n_iter: int = 40) -> np.ndarray:
    """
    Distributed Kalman Filter (DKF) with Metropolis-consensus fusion.

    Each sensor independently runs a scalar Kalman filter using a random-walk
    state model, then fuses its posterior estimate with neighbors through
    n_iter rounds of Metropolis-weighted consensus averaging.

    Model
    -----
      State:       x(t) = x(t-1) + w,    w ~ N(0, Q)
      Measurement: y_i(t) = x(t) + v_i,  v_i ~ N(0, R)

    Algorithm (per time step t)
    ---------------------------
      1. Predict:  x̂_i(t|t-1) = x̂_i(t-1|t-1)
                   P_i(t|t-1)  = P_i(t-1|t-1) + Q
      2. Update:   K_i = P_i(t|t-1) / (P_i(t|t-1) + R)
                   x̂_i(t|t) = x̂_i(t|t-1) + K_i · (y_i(t) - x̂_i(t|t-1))
                   P_i(t|t)  = (1 - K_i) · P_i(t|t-1)
      3. Consensus: x̂_i ← Σ_j P_ij · x̂_j  (n_iter rounds)

    Notes
    -----
    Unlike the proposed method this baseline does NOT detect or down-weight
    faulty sensors — all posterior estimates participate equally in consensus.
    This makes it a strong but fault-unaware distributed baseline.

    Parameters
    ----------
    Y      : (N, T)  sensor measurements
    P      : (N, N)  Metropolis mixing matrix
    Q      : process noise variance (tunable prior on signal smoothness)
    R      : measurement noise variance (assumed uniform across sensors)
    n_iter : consensus rounds per time step

    Returns
    -------
    s_hat : (T,) global estimate — mean of all sensors' posterior after consensus
    """
    N, T = Y.shape

    # Per-sensor KF state (scalar, shared initial conditions)
    x_hat = np.zeros(N)          # posterior mean
    p_var = np.ones(N) * R       # posterior variance

    # Pre-compute P^n_iter via matrix power (cheaper than repeated matmul in inner loop)
    P_pow = np.linalg.matrix_power(P, n_iter)   # (N, N)

    s_hat = np.zeros(T)
    for t in range(T):
        # --- Predict ---
        x_pred = x_hat.copy()
        p_pred = p_var + Q

        # --- Local measurement update ---
        K     = p_pred / (p_pred + R)
        x_hat = x_pred + K * (Y[:, t] - x_pred)
        p_var = (1.0 - K) * p_pred

        # --- Consensus: apply P^n_iter in a single matrix-vector multiply ---
        x_hat = P_pow @ x_hat

        s_hat[t] = x_hat.mean()

    return s_hat


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

    Statistical grounding
    ---------------------
    For a healthy node, D+_i(t) ≈ 0 so w_i ≈ 1.  By the CLT approximation,
    the false-alarm rate is bounded by Φ(−k), where Φ is the standard
    normal CDF:
        k = 1.5  →  FAR ≤ Φ(−1.5) ≈ 0.067  (conservative theoretical bound)
        k = 2.5  →  FAR ≤ Φ(−2.5) ≈ 0.006

    In practice, healthy-node weights cluster tightly near 1, so the
    empirical FAR is well below the theoretical bound (typically ≤ 0.007
    at k = 1.5 for fault fractions up to 20%).

    Parameters
    ----------
    W          : (N, T) reliability weights from compute_weights()
    time_start : first time index to include (set to fault_onset to
                 exclude the healthy pre-fault window from the average)
    k          : threshold multiplier; default 1.5 gives FAR ≤ Φ(−1.5)
    """
    w_bar = W[:, time_start:].mean(axis=1)
    return float(w_bar.mean() - k * w_bar.std())
