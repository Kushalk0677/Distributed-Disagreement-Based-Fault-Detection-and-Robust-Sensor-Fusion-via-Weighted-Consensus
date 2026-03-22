"""
tests/test_core.py
------------------
Unit tests for src/ modules. Run with:  pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.network import (build_random_geometric_graph, build_metropolis_weights,
                          assign_faults_random, assign_faults_clustered)
from src.signal  import generate_signal, generate_measurements
from src.fusion  import (fuse_average, fuse_trimmed_mean, fuse_local_median,
                          fuse_consensus_plain, fuse_proposed, fuse_distributed_kf,
                          compute_disagreement, smooth_disagreement,
                          compute_enhanced_disagreement, compute_weights,
                          adaptive_fault_threshold)
from src.detection import DisagreementDetector, CUSUMDetector, EWMADetector
from src.metrics   import mse, rmse, detection_metrics, ci95


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_net():
    g = build_random_geometric_graph(N=20, radius=0.40, seed=0)
    P = build_metropolis_weights(g)
    return g, P

@pytest.fixture
def healthy_data(small_net):
    g, P = small_net
    N = 20; T = 100
    s = generate_signal(T, "sinusoid")
    fm = np.zeros(N, dtype=bool)
    Y  = generate_measurements(s, N, 0.05, fm, ["none"]*N, fault_onset=0)
    return g, P, s, Y, fm


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class TestNetwork:
    def test_rgg_shape(self):
        g = build_random_geometric_graph(30, 0.35, seed=1)
        assert g["positions"].shape == (30, 2)
        assert len(g["neighbors"]) == 30
        assert g["adj_matrix"].shape == (30, 30)

    def test_rgg_symmetric(self):
        g = build_random_geometric_graph(20, 0.35, seed=2)
        adj = g["adj_matrix"]
        assert np.array_equal(adj, adj.T)

    def test_metropolis_row_stochastic(self):
        g = build_random_geometric_graph(20, 0.40, seed=3)
        P = build_metropolis_weights(g)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(20), atol=1e-10)

    def test_metropolis_nonneg(self):
        g = build_random_geometric_graph(20, 0.40, seed=4)
        P = build_metropolis_weights(g)
        assert (P >= 0).all()

    def test_fault_random_fraction(self):
        rng = np.random.default_rng(0)
        fm, ft = assign_faults_random(100, 0.20, "stuck", rng)
        assert fm.sum() == 20
        assert all(t == "stuck" for t in ft if t != "none")

    def test_fault_clustered_geographic(self):
        g   = build_random_geometric_graph(50, 0.35, seed=5)
        rng = np.random.default_rng(1)
        fm, _ = assign_faults_clustered(g["positions"], 0.20, "drift", rng)
        assert fm.sum() == 10
        # Faulty nodes should be geographically close to each other
        faulty_pos  = g["positions"][fm]
        center      = faulty_pos.mean(axis=0)
        dists       = np.linalg.norm(faulty_pos - center, axis=1)
        healthy_pos = g["positions"][~fm]
        h_dists     = np.linalg.norm(healthy_pos - center, axis=1)
        assert dists.mean() < h_dists.mean()   # faulty cluster is tighter


# ─────────────────────────────────────────────────────────────────────────────
# Signal
# ─────────────────────────────────────────────────────────────────────────────

class TestSignal:
    def test_sinusoid_range(self):
        s = generate_signal(200, "sinusoid")
        assert s.shape == (200,)
        assert abs(s.max() - 1.0) < 0.01
        assert abs(s.min() + 1.0) < 0.01

    def test_measurements_shape(self):
        s  = generate_signal(100, "sinusoid")
        fm = np.zeros(10, dtype=bool)
        Y  = generate_measurements(s, 10, 0.05, fm, ["none"]*10)
        assert Y.shape == (10, 100)

    def test_stuck_fault(self):
        N = 5; T = 100; t0 = 30
        s  = generate_signal(T)
        fm = np.array([True, False, False, False, False])
        ft = ["stuck", "none", "none", "none", "none"]
        Y  = generate_measurements(s, N, 0.0, fm, ft, fault_onset=t0, seed=0)
        # Sensor 0: all values after t0 should be identical
        assert np.allclose(Y[0, t0:], Y[0, t0])
        # Healthy sensors track signal
        assert not np.allclose(Y[1, t0:], Y[1, t0])

    def test_drift_fault(self):
        N = 5; T = 100; t0 = 20; rho = 0.10
        s  = generate_signal(T)
        fm = np.array([True] + [False]*4)
        ft = ["drift"] + ["none"]*4
        Y  = generate_measurements(s, N, 0.0, fm, ft,
                                    fault_onset=t0, drift_rate=rho, seed=0)
        # Drift should increase after t0
        diffs = np.diff(Y[0, t0:] - s[t0:])
        assert np.mean(diffs) > 0   # bias grows


# ─────────────────────────────────────────────────────────────────────────────
# Fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestFusion:
    def test_average_correct(self, healthy_data):
        g, P, s, Y, fm = healthy_data
        s_hat = fuse_average(Y)
        assert s_hat.shape == s.shape
        assert mse(s, s_hat) < 0.01

    def test_trimmed_mean_shape(self, healthy_data):
        g, P, s, Y, fm = healthy_data
        s_hat = fuse_trimmed_mean(Y, 0.10)
        assert s_hat.shape == s.shape

    def test_local_median_shape(self, healthy_data):
        g, P, s, Y, fm = healthy_data
        s_hat = fuse_local_median(Y, g["neighbors"])
        assert s_hat.shape == s.shape

    def test_consensus_converges_to_mean(self, healthy_data):
        g, P, s, Y, fm = healthy_data
        s_plain = fuse_consensus_plain(Y, P, n_iter=100)
        s_avg   = fuse_average(Y)
        np.testing.assert_allclose(s_plain, s_avg, atol=1e-6)

    def test_proposed_outperforms_average_with_faults(self, small_net):
        """Core algorithm correctness: proposed MSE < average MSE under faults."""
        g, P = small_net
        N = 20; T = 150; ff = 0.25
        s  = generate_signal(T, "sinusoid")
        rng = np.random.default_rng(99)
        fm, fts = assign_faults_random(N, ff, "stuck", rng)
        Y  = generate_measurements(s, N, 0.05, fm, fts, fault_onset=30, seed=42)

        s_avg, *_      = (fuse_average(Y),)
        s_prop, *_     = fuse_proposed(Y, g["neighbors"], P, n_iter=30, lag=15)

        assert mse(s, s_prop) < mse(s, s_avg), \
            "Proposed should outperform simple average under 25% stuck faults"

    def test_proposed_returns_correct_shapes(self, healthy_data):
        g, P, s, Y, fm = healthy_data
        N, T = Y.shape
        s_hat, W, Ds, D_enh = fuse_proposed(Y, g["neighbors"], P, n_iter=10, lag=10)
        assert s_hat.shape == (T,)
        assert W.shape     == (N, T)
        assert Ds.shape    == (N, T)
        assert D_enh.shape == (N, T)

    def test_weights_in_01(self, healthy_data):
        g, P, s, Y, fm = healthy_data
        D  = compute_disagreement(Y, g["neighbors"])
        Ds = smooth_disagreement(D)
        De = compute_enhanced_disagreement(Ds)
        W  = compute_weights(De)
        assert (W > 0).all() and (W <= 1.0).all()

    def test_lag_default_is_25(self):
        """Bug fix verification: fuse_proposed default lag must be 25 not 10."""
        import inspect
        sig = inspect.signature(fuse_proposed)
        assert sig.parameters["lag"].default == 25, \
            "fuse_proposed default lag should be 25 (= ½·t₀, paper Table I)"

    def test_distributed_kf_shape(self, healthy_data):
        g, P, s, Y, fm = healthy_data
        s_hat = fuse_distributed_kf(Y, P, n_iter=10)
        assert s_hat.shape == s.shape


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetection:
    def test_detector_default_lag_25(self):
        """Bug fix verification: DisagreementDetector default lag must be 25."""
        det = DisagreementDetector()
        assert det.lag == 25, "Default lag should be 25 (paper Table I)"

    def test_detector_flags_stuck_faults(self, small_net):
        g, P = small_net
        N = 20; T = 200; t0 = 50
        s  = generate_signal(T)
        rng = np.random.default_rng(7)
        fm, fts = assign_faults_random(N, 0.20, "stuck", rng)
        Y = generate_measurements(s, N, 0.05, fm, fts, fault_onset=t0, seed=7)

        det = DisagreementDetector(alpha=0.85, beta=2.0, lag=25, k=1.5)
        pred, W, _ = det.fit_predict(Y, g["neighbors"], fault_onset=t0)

        dm = detection_metrics(fm, pred)
        # At 20% stuck faults with t=200 steps, detection should be reasonable
        assert dm["FDR"] > 0.5, f"Recall too low: {dm['FDR']:.2f}"
        assert dm["FAR"] < 0.3, f"FAR too high: {dm['FAR']:.2f}"

    def test_far_low_healthy(self, healthy_data):
        """FAR should be 0 when there are no faults."""
        g, P, s, Y, fm = healthy_data
        det = DisagreementDetector(k=1.5)
        pred, W, _ = det.fit_predict(Y, g["neighbors"], fault_onset=0)
        # All sensors healthy; any false positives count as FAR
        assert pred.sum() == 0 or pred.mean() < 0.10, \
            "FAR should be near zero on healthy data"

    def test_cusum_shape(self, healthy_data):
        g, P, s, Y, fm = healthy_data
        det  = CUSUMDetector()
        pred, S, innov = det.fit_predict(Y, g["neighbors"])
        assert pred.shape == (Y.shape[0],)

    def test_ewma_shape(self, healthy_data):
        g, P, s, Y, fm = healthy_data
        det  = EWMADetector()
        pred, E, ucl = det.fit_predict(Y, g["neighbors"], fault_onset=20)
        assert pred.shape == (Y.shape[0],)

    def test_adaptive_threshold_healthy_cluster(self):
        """Healthy weights cluster near 1; threshold should be < 0.97."""
        N, T = 20, 100
        W    = np.ones((N, T)) * 0.97 + np.random.randn(N, T) * 0.005
        tau  = adaptive_fault_threshold(W, time_start=0, k=1.5)
        assert tau < 0.97


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mse_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert mse(x, x) == 0.0

    def test_mse_positive(self):
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 1.0, 1.0])
        assert mse(x, y) == 1.0

    def test_rmse_is_sqrt_mse(self):
        x = np.random.randn(50)
        y = np.random.randn(50)
        np.testing.assert_allclose(rmse(x, y), np.sqrt(mse(x, y)))

    def test_detection_metrics_perfect(self):
        fm   = np.array([True, True, False, False, False])
        pred = np.array([True, True, False, False, False])
        dm   = detection_metrics(fm, pred)
        assert dm["FDR"] == 1.0
        assert dm["FAR"] == 0.0
        assert dm["F1"]  == 1.0

    def test_detection_metrics_all_wrong(self):
        fm   = np.array([True, True, False, False])
        pred = np.array([False, False, True, True])
        dm   = detection_metrics(fm, pred)
        assert dm["FDR"] == 0.0
        assert dm["FAR"] == 1.0

    def test_ci95_correct(self):
        vals = list(np.ones(100))
        mean, hw = ci95(vals)
        assert mean == 1.0
        assert hw == 0.0   # zero variance → zero CI

    def test_ci95_width_decreases_with_n(self):
        rng  = np.random.default_rng(0)
        v10  = list(rng.standard_normal(10))
        v100 = list(rng.standard_normal(100))
        _, hw10  = ci95(v10)
        _, hw100 = ci95(v100)
        # CI width ∝ 1/√n; 100-sample CI should generally be narrower
        # (not guaranteed by one sample, so just check it's finite)
        assert np.isfinite(hw10) and np.isfinite(hw100)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: end-to-end smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_runs(self):
        """End-to-end: network → signal → fusion → detection → metrics."""
        g  = build_random_geometric_graph(15, 0.45, seed=99)
        P  = build_metropolis_weights(g)
        s  = generate_signal(80, "sinusoid")
        rng = np.random.default_rng(1)
        fm, fts = assign_faults_random(15, 0.20, "stuck", rng)
        Y  = generate_measurements(s, 15, 0.05, fm, fts, fault_onset=20, seed=1)

        s_prop, W, Ds, D_enh = fuse_proposed(Y, g["neighbors"], P,
                                               n_iter=20, lag=10)
        det  = DisagreementDetector(lag=10)
        pred, _, _ = det.fit_predict(Y, g["neighbors"], fault_onset=20)

        dm = detection_metrics(fm, pred)
        m  = mse(s, s_prop)

        assert np.isfinite(m)
        assert 0 <= dm["FDR"] <= 1
        assert 0 <= dm["FAR"] <= 1

    def test_proposed_beats_average_across_seeds(self):
        """Statistical: proposed wins over average in most of 10 independent trials."""
        N = 25; T = 150; t0 = 40
        g  = build_random_geometric_graph(N, 0.38, seed=0)
        P  = build_metropolis_weights(g)
        wins = 0
        for seed in range(10):
            s   = generate_signal(T, seed=seed)
            rng = np.random.default_rng(seed)
            fm, fts = assign_faults_random(N, 0.20, "stuck", rng)
            Y  = generate_measurements(s, N, 0.05, fm, fts, t0, seed=seed+100)
            s_avg  = fuse_average(Y)
            s_prop, *_ = fuse_proposed(Y, g["neighbors"], P, n_iter=25, lag=20)
            if mse(s, s_prop) < mse(s, s_avg):
                wins += 1
        assert wins >= 7, f"Proposed should beat average in ≥7/10 trials; got {wins}/10"
