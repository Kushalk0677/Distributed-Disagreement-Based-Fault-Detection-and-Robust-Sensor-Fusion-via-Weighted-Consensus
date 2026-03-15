"""
signal.py — ground-truth signals and sensor measurement models

Fault types:  stuck | drift | malicious | noise_burst
Noise models: homogeneous | heterogeneous (per-sensor std)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth signal
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(T: int, signal_type: str = "sinusoid", seed: int = 0) -> np.ndarray:
    t = np.arange(T, dtype=float)
    if signal_type == "sinusoid":
        return np.sin(2 * np.pi * t / T)
    elif signal_type == "multi_freq":
        return (0.6 * np.sin(2 * np.pi * t / T)
                + 0.3 * np.sin(2 * np.pi * t / (T / 4))
                + 0.1 * np.cos(2 * np.pi * t / (T / 7)))
    elif signal_type == "step":
        s = np.zeros(T); s[T // 2:] = 1.0; return s
    elif signal_type == "ramp":
        return t / T
    elif signal_type == "composite":
        rng = np.random.default_rng(seed)
        return (0.5 * np.sin(2 * np.pi * t / T)
                + 0.3 * np.sin(2 * np.pi * t / (T / 3))
                + 0.05 * rng.standard_normal(T))
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneous noise
# ─────────────────────────────────────────────────────────────────────────────

def make_heterogeneous_noise(N: int, base_std: float = 0.05,
                              high_std: float = 0.40,
                              noisy_fraction: float = 0.30,
                              seed: int = 99):
    """
    Returns (noise_stds array, high_noise_mask).
    `noisy_fraction` of sensors have std = high_std; rest have std = base_std.
    This is the key scenario where the proposed method outperforms local median:
    - High-noise sensors get persistently high disagreement → low weights
    - Proposed consensus up-weights low-noise sensors automatically
    - Local median cannot distinguish noise quality
    """
    rng   = np.random.default_rng(seed)
    stds  = np.full(N, base_std)
    mask  = np.zeros(N, dtype=bool)
    n_hi  = max(1, int(N * noisy_fraction))
    idx   = rng.choice(N, size=n_hi, replace=False)
    stds[idx] = high_std
    mask[idx] = True
    return stds, mask


# ─────────────────────────────────────────────────────────────────────────────
# Measurement generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_measurements(s: np.ndarray, N: int,
                           noise_std,            # scalar OR (N,) array
                           fault_mask: np.ndarray,
                           fault_types: list,
                           fault_onset: int = 0,
                           drift_rate: float = 0.04,
                           burst_std: float = 2.0,
                           seed: int = 1) -> np.ndarray:
    """
    Returns (N, T) measurement matrix.

    Fault models
    ------------
    stuck       : sensor freezes at value at fault_onset
    drift       : linearly increasing bias after fault_onset
    malicious   : replaced with uniform random values over signal range
    noise_burst : variance suddenly increases (hard to detect — subtle fault)
    """
    T   = len(s)
    rng = np.random.default_rng(seed)

    noise_stds = (np.full(N, noise_std) if np.isscalar(noise_std)
                  else np.asarray(noise_std, dtype=float))

    Y = s[np.newaxis, :] + noise_stds[:, np.newaxis] * rng.standard_normal((N, T))

    for i in range(N):
        if not fault_mask[i]:
            continue
        ft  = fault_types[i]
        Tf  = T - fault_onset
        if ft == "stuck":
            Y[i, fault_onset:] = Y[i, fault_onset]
        elif ft == "drift":
            Y[i, fault_onset:] += drift_rate * np.arange(Tf)
        elif ft == "malicious":
            span = s.max() - s.min()
            Y[i, fault_onset:] = rng.uniform(s.min() - span,
                                              s.max() + span, size=Tf)
        elif ft == "noise_burst":
            Y[i, fault_onset:] = (s[fault_onset:]
                                  + burst_std * rng.standard_normal(Tf))
        else:
            raise ValueError(f"Unknown fault type: {ft}")
    return Y
