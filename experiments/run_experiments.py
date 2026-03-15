#!/usr/bin/env python3
"""
run_experiments.py
------------------

Usage
-----
  python experiments/run_experiments.py              # full (~15 min)
  python experiments/run_experiments.py --quick      # ~1 min
"""

import sys, argparse, json
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network   import (build_random_geometric_graph,
                            assign_faults_random, assign_faults_clustered,
                            build_metropolis_weights,
                            assign_faults_random, assign_faults_clustered)
from src.signal    import (generate_signal, generate_measurements,
                            make_heterogeneous_noise)
from src.fusion    import (fuse_average, fuse_trimmed_mean, fuse_local_median,
                            fuse_consensus_plain, fuse_proposed)
from src.detection import DisagreementDetector, CUSUMDetector, EWMADetector
from src.metrics   import mse, rmse, detection_metrics, communication_overhead, ci95
import src.visualize as viz


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_net(cfg, seed_offset=0):
    net = cfg["network"]
    g   = build_random_geometric_graph(net["N"], net["radius"],
                                        seed=net["seed"] + seed_offset)
    P   = build_metropolis_weights(g)
    return g, P


def run_one(cfg, fault_fraction, fault_type,
            noise_std_arr,          # scalar or (N,) array
            graph, P,
            sig_seed, fault_seed,
            clustered=False):
    """Single simulation trial. Returns dict of estimates + metrics."""
    sig = cfg["signal"]
    flt = cfg["faults"]
    fus = cfg["fusion"]
    det = cfg["detection"]
    N   = len(graph["neighbors"])

    s = generate_signal(sig["T"], signal_type=sig["type"], seed=sig_seed)

    rng = np.random.default_rng(fault_seed)
    if fault_fraction == 0.0:
        fault_mask  = np.zeros(N, dtype=bool)
        fault_types = ["none"] * N
    elif clustered:
        fault_mask, fault_types = assign_faults_clustered(
            graph["positions"], fault_fraction, fault_type, rng)
    else:
        fault_mask, fault_types = assign_faults_random(
            N, fault_fraction, fault_type, rng)

    Y = generate_measurements(
        s, N, noise_std_arr,
        fault_mask=fault_mask, fault_types=fault_types,
        fault_onset=flt["fault_onset"], drift_rate=flt["drift_rate"],
        burst_std=flt["burst_std"], seed=fault_seed + 100,
    )

    # ── All fusion methods ──────────────────────────────────────────────────
    s_avg  = fuse_average(Y)
    s_tri  = fuse_trimmed_mean(Y, trim=fus["trim"])
    s_lmed = fuse_local_median(Y, graph["neighbors"])
    s_pcon = fuse_consensus_plain(Y, P, n_iter=fus["n_iter"])
    s_prop, W, Ds, D_enh = fuse_proposed(
        Y, graph["neighbors"], P,
        alpha=fus["alpha"], beta=fus["beta"],
        lag=fus["lag"], n_iter=fus["n_iter"],
    )

    # ── Detectors ───────────────────────────────────────────────────────────
    dd  = DisagreementDetector(alpha=fus["alpha"], beta=fus["beta"],
                                lag=fus["lag"],   k=fus["k_threshold"])
    cd  = CUSUMDetector(slack=det["cusum_slack"], threshold=det["cusum_threshold"])
    ed  = EWMADetector(lam=det["ewma_lambda"], L=det["ewma_L"])

    pred_dd, _, _       = dd.fit_predict(Y, graph["neighbors"], flt["fault_onset"])
    pred_cd, _, _       = cd.fit_predict(Y, graph["neighbors"], flt["fault_onset"])
    pred_ed, _, _       = ed.fit_predict(Y, graph["neighbors"], flt["fault_onset"])

    dm_dd = detection_metrics(fault_mask, pred_dd)
    dm_cd = detection_metrics(fault_mask, pred_cd)
    dm_ed = detection_metrics(fault_mask, pred_ed)

    estimates = dict(
        Average=s_avg, **{"Trimmed Mean": s_tri},
        **{"Local Median": s_lmed},
        **{"Plain Consensus": s_pcon},
        Proposed=s_prop,
    )

    return dict(
        s_true=s, estimates=estimates,
        W=W, Ds=Ds, D_enh=D_enh,
        fault_mask=fault_mask, fault_onset=flt["fault_onset"],
        mse={k: mse(s, v) for k, v in estimates.items()},
        det=dict(Proposed=dm_dd, CUSUM=dm_cd, EWMA=dm_ed),
    )


def mc(cfg, n_trials, fault_fraction, fault_type,
       noise_std=None, clustered=False, net_seed_var=False):
    """Monte Carlo: n_trials independent runs."""
    net = cfg["network"]
    g0, P0 = build_net(cfg)
    if noise_std is None:
        noise_std = cfg["signal"]["noise_std"]
    rows = []
    for t in range(n_trials):
        g, P = build_net(cfg, seed_offset=t) if net_seed_var else (g0, P0)
        rows.append(run_one(cfg, fault_fraction, fault_type,
                             noise_std, g, P,
                             sig_seed=t, fault_seed=t * 7,
                             clustered=clustered))
    return rows


def mean_ci(rows, key, subkey=None):
    if subkey:
        vals = [r[key][subkey] for r in rows]
    else:
        vals = [r[key] for r in rows]
    return ci95(vals)


def mse_mean_ci(rows, method):
    vals = [r["mse"][method] for r in rows]
    return ci95(vals)


def det_mean(rows, detector, metric):
    return float(np.mean([r["det"][detector][metric] for r in rows]))


METHODS = ["Average", "Trimmed Mean", "Local Median", "Plain Consensus", "Proposed"]


# ─────────────────────────────────────────────────────────────────────────────
# Exp 1: Single-run illustration
# ─────────────────────────────────────────────────────────────────────────────

def exp1_illustration(cfg):
    print("\n=== Exp 1: Single-Run Illustration ===")
    g, P = build_net(cfg)
    r = run_one(cfg, cfg["faults"]["fault_fraction"], cfg["faults"]["fault_type"],
                cfg["signal"]["noise_std"], g, P, sig_seed=0, fault_seed=10)

    viz.plot_topology(g, r["fault_mask"])
    viz.plot_estimation(np.arange(cfg["signal"]["T"]), r["s_true"],
                        r["estimates"], r["fault_onset"])
    viz.plot_weights(r["W"], r["fault_mask"], r["fault_onset"])

    print("  MSE:", {k: f"{v:.5f}" for k, v in r["mse"].items()})
    det_str = {d: f"FDR={r['det'][d]['FDR']:.2f} FAR={r['det'][d]['FAR']:.2f} F1={r['det'][d]['F1']:.2f}"
               for d in ["Proposed", "CUSUM", "EWMA"]}
    print("  Detection:", det_str)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Exp 2: Enhanced vs raw disagreement (drift fault)
# ─────────────────────────────────────────────────────────────────────────────

def exp2_enhanced_disagreement(cfg):
    print("\n=== Exp 2: Enhanced Disagreement (Drift) ===")
    g, P = build_net(cfg)
    r = run_one(cfg, 0.20, "drift", cfg["signal"]["noise_std"],
                g, P, sig_seed=0, fault_seed=42)
    viz.plot_enhanced_vs_raw(r["Ds"], r["D_enh"], r["fault_mask"], r["fault_onset"])
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Exp 3: MSE vs fault fraction — all 5 methods
# ─────────────────────────────────────────────────────────────────────────────

def exp3_mse_vs_fault_fraction(cfg):
    print("\n=== Exp 3: MSE vs Fault Fraction ===")
    fracs, n = cfg["sweeps"]["fault_fractions"], cfg["sweeps"]["n_trials"]
    ft = cfg["faults"]["fault_type"]

    mse_means = {m: [] for m in METHODS}
    mse_cis   = {m: [] for m in METHODS}

    for ff in fracs:
        rows = mc(cfg, n, ff, ft)
        for m in METHODS:
            mu, ci = mse_mean_ci(rows, m)
            mse_means[m].append(mu)
            mse_cis[m].append(ci)
        print(f"  {ff:.0%}  " + "  ".join(
            f"{m[:4]}={mse_means[m][-1]:.5f}" for m in METHODS))

    viz.plot_mse_vs_fault_fraction(fracs, mse_means, mse_cis)
    return dict(fractions=fracs, mse=mse_means, ci=mse_cis)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 4: Detector comparison — FDR / FAR / F1
# ─────────────────────────────────────────────────────────────────────────────

def exp4_detector_comparison(cfg):
    print("\n=== Exp 4: Detector Comparison ===")
    fracs, n = cfg["sweeps"]["fault_fractions"], cfg["sweeps"]["n_trials"]
    ft = cfg["faults"]["fault_type"]
    DETS = ["Proposed", "CUSUM", "EWMA"]

    det_results = {d: {"fdr": [], "far": [], "f1": []} for d in DETS}
    for ff in fracs:
        rows = mc(cfg, n, ff, ft)
        for d in DETS:
            det_results[d]["fdr"].append(det_mean(rows, d, "FDR"))
            det_results[d]["far"].append(det_mean(rows, d, "FAR"))
            det_results[d]["f1"].append(det_mean(rows, d, "F1"))
        print(f"  {ff:.0%}  Prop: FDR={det_results['Proposed']['fdr'][-1]:.2f}"
              f" F1={det_results['Proposed']['f1'][-1]:.2f} |"
              f" CUSUM: FDR={det_results['CUSUM']['fdr'][-1]:.2f}"
              f" F1={det_results['CUSUM']['f1'][-1]:.2f} |"
              f" EWMA: FDR={det_results['EWMA']['fdr'][-1]:.2f}"
              f" F1={det_results['EWMA']['f1'][-1]:.2f}")
    viz.plot_detector_comparison(fracs, det_results)
    return dict(fractions=fracs, det=det_results)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 5: Clustered faults — proposed beats local median
# ─────────────────────────────────────────────────────────────────────────────

def exp5_clustered_faults(cfg):
    """
    Key result: proposed outperforms local median when faults are geographically
    clustered (some sensors have >50% faulty neighbors, breaking local median).
    For random fault placement, local median is competitive on MSE, but proposed
    additionally provides fault detection capability.
    """
    print("\n=== Exp 5: Clustered vs Random Faults ===")
    fracs, n = cfg["sweeps"]["fault_fractions"], cfg["sweeps"]["n_trials"]
    ft = cfg["faults"]["fault_type"]
    ns = cfg["signal"]["noise_std"]

    g, P = build_net(cfg)

    res_rand_lm, res_rand_pr = [], []
    res_clus_lm, res_clus_pr = [], []

    for ff in fracs:
        rows_rand, rows_clus = [], []
        for t in range(n):
            s = generate_signal(cfg["signal"]["T"], cfg["signal"]["type"], seed=t)
            flt = cfg["faults"]

            rng_r = np.random.default_rng(t * 7)
            fm_r, fts_r = assign_faults_random(cfg["network"]["N"], ff, ft, rng_r)
            Y_r = generate_measurements(s, cfg["network"]["N"], ns, fm_r, fts_r,
                                         flt["fault_onset"], flt["drift_rate"],
                                         flt["burst_std"], seed=t + 100)

            rng_c = np.random.default_rng(t * 7)
            fm_c, fts_c = assign_faults_clustered(
                g["positions"], ff, ft, rng_c)
            Y_c = generate_measurements(s, cfg["network"]["N"], ns, fm_c, fts_c,
                                         flt["fault_onset"], flt["drift_rate"],
                                         flt["burst_std"], seed=t + 100)

            fus = cfg["fusion"]
            lm_r = fuse_local_median(Y_r, g["neighbors"])
            pr_r, *_ = fuse_proposed(Y_r, g["neighbors"], P,
                                      alpha=fus["alpha"], beta=fus["beta"],
                                      lag=fus["lag"], n_iter=fus["n_iter"])
            lm_c = fuse_local_median(Y_c, g["neighbors"])
            pr_c, *_ = fuse_proposed(Y_c, g["neighbors"], P,
                                      alpha=fus["alpha"], beta=fus["beta"],
                                      lag=fus["lag"], n_iter=fus["n_iter"])
            rows_rand.append((mse(s, lm_r), mse(s, pr_r)))
            rows_clus.append((mse(s, lm_c), mse(s, pr_c)))

        res_rand_lm.append(float(np.mean([r[0] for r in rows_rand])))
        res_rand_pr.append(float(np.mean([r[1] for r in rows_rand])))
        res_clus_lm.append(float(np.mean([r[0] for r in rows_clus])))
        res_clus_pr.append(float(np.mean([r[1] for r in rows_clus])))

        print(f"  {ff:.0%}  RAND: lmed={res_rand_lm[-1]:.5f} prop={res_rand_pr[-1]:.5f}"
              f"  CLUST: lmed={res_clus_lm[-1]:.5f} prop={res_clus_pr[-1]:.5f}"
              f"  prop<lmed(clust)={res_clus_pr[-1] < res_clus_lm[-1]}")

    # Topology comparison figure
    rng_r = np.random.default_rng(42)
    fm_r, _ = assign_faults_random(cfg["network"]["N"], 0.25, ft, rng_r)
    rng_c = np.random.default_rng(42)
    fm_c, _ = assign_faults_clustered(g["positions"], 0.25, ft, rng_c)
    viz.plot_clustered_topology(g, fm_r, fm_c)

    results = {
        "Proposed (random)":        res_rand_pr,
        "Local Median (random)":    res_rand_lm,
        "Proposed (clustered)":     res_clus_pr,
        "Local Median (clustered)": res_clus_lm,
    }
    viz.plot_clustered_vs_random(fracs, results)
    return dict(fractions=fracs, results=results)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 6: Alpha sweep
# ─────────────────────────────────────────────────────────────────────────────

def exp6_alpha_sweep(cfg):
    print("\n=== Exp 6: Alpha Sweep ===")
    alphas, n = cfg["sweeps"]["alphas"], cfg["sweeps"]["n_trials"]
    flt, fus  = cfg["faults"], cfg["fusion"]
    g, P      = build_net(cfg)
    ns        = cfg["signal"]["noise_std"]

    mse_vals = []
    for alpha in alphas:
        rows = []
        for t in range(n):
            s   = generate_signal(cfg["signal"]["T"], cfg["signal"]["type"], seed=t)
            rng = np.random.default_rng(t * 7)
            fm, fts = assign_faults_random(
                cfg["network"]["N"], flt["fault_fraction"], flt["fault_type"], rng)
            Y = generate_measurements(s, cfg["network"]["N"], ns,
                                       fm, fts, flt["fault_onset"],
                                       flt["drift_rate"], flt["burst_std"],
                                       seed=t * 7 + 100)
            sp, *_ = fuse_proposed(Y, g["neighbors"], P, alpha=alpha,
                                    beta=fus["beta"], lag=fus["lag"],
                                    n_iter=fus["n_iter"])
            rows.append(mse(s, sp))
        mse_vals.append(float(np.mean(rows)))
        print(f"  α={alpha:.2f}  MSE={mse_vals[-1]:.6f}")
    viz.plot_mse_vs_alpha(alphas, mse_vals)
    return dict(alphas=alphas, mse_prop=mse_vals)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 7: 2-D sensitivity heatmap (beta × lag)
# ─────────────────────────────────────────────────────────────────────────────

def exp7_sensitivity_heatmap(cfg):
    print("\n=== Exp 7: Beta × Lag Sensitivity Heatmap ===")
    betas, lags = cfg["sweeps"]["betas"], cfg["sweeps"]["lags"]
    n = max(10, cfg["sweeps"]["n_trials"] // 3)   # fewer trials for speed
    flt, fus = cfg["faults"], cfg["fusion"]
    g, P     = build_net(cfg)
    ns       = cfg["signal"]["noise_std"]

    mse_grid = np.zeros((len(betas), len(lags)))
    for bi, beta in enumerate(betas):
        for li, lag in enumerate(lags):
            trial_mse = []
            for t in range(n):
                s   = generate_signal(cfg["signal"]["T"], cfg["signal"]["type"], seed=t)
                rng = np.random.default_rng(t * 7)
                fm, fts = assign_faults_random(
                    cfg["network"]["N"], flt["fault_fraction"], flt["fault_type"], rng)
                Y   = generate_measurements(s, cfg["network"]["N"], ns,
                                             fm, fts, flt["fault_onset"],
                                             flt["drift_rate"], flt["burst_std"],
                                             seed=t * 7 + 100)
                sp, *_ = fuse_proposed(Y, g["neighbors"], P,
                                        alpha=fus["alpha"], beta=beta,
                                        lag=lag, n_iter=fus["n_iter"])
                trial_mse.append(mse(s, sp))
            mse_grid[bi, li] = np.mean(trial_mse)
        print(f"  β={beta:.1f}: {[f'{mse_grid[bi,li]:.5f}' for li in range(len(lags))]}")

    viz.plot_sensitivity_heatmap(betas, lags, mse_grid)
    return dict(betas=betas, lags=lags, mse_grid=mse_grid.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Exp 8: Consensus convergence speed
# ─────────────────────────────────────────────────────────────────────────────

def exp8_consensus_convergence(cfg):
    print("\n=== Exp 8: Consensus Convergence Speed ===")
    n_iter_vals = cfg["sweeps"]["n_iter_vals"]
    n = max(10, cfg["sweeps"]["n_trials"] // 3)
    flt, fus = cfg["faults"], cfg["fusion"]
    g, P     = build_net(cfg)
    ns       = cfg["signal"]["noise_std"]

    mse_prop, mse_plain = [], []
    for K in n_iter_vals:
        tp, tc = [], []
        for t in range(n):
            s   = generate_signal(cfg["signal"]["T"], cfg["signal"]["type"], seed=t)
            rng = np.random.default_rng(t * 7)
            fm, fts = assign_faults_random(
                cfg["network"]["N"], flt["fault_fraction"], flt["fault_type"], rng)
            Y   = generate_measurements(s, cfg["network"]["N"], ns,
                                         fm, fts, flt["fault_onset"],
                                         flt["drift_rate"], flt["burst_std"],
                                         seed=t * 7 + 100)
            sp, *_ = fuse_proposed(Y, g["neighbors"], P,
                                    alpha=fus["alpha"], beta=fus["beta"],
                                    lag=fus["lag"], n_iter=K)
            sc = fuse_consensus_plain(Y, P, n_iter=K)
            tp.append(mse(s, sp))
            tc.append(mse(s, sc))
        mse_prop.append(float(np.mean(tp)))
        mse_plain.append(float(np.mean(tc)))
        print(f"  K={K:3d}  proposed={mse_prop[-1]:.5f}  plain={mse_plain[-1]:.5f}")

    viz.plot_consensus_convergence(n_iter_vals, mse_prop, mse_plain)
    return dict(n_iter_vals=n_iter_vals, mse_proposed=mse_prop, mse_plain=mse_plain)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 9: Network size scaling
# ─────────────────────────────────────────────────────────────────────────────

def exp9_network_size(cfg):
    print("\n=== Exp 9: Network Size Scaling ===")
    N_vals, n = cfg["sweeps"]["N_values"], cfg["sweeps"]["n_trials"]
    ft  = cfg["faults"]["fault_type"]
    ff  = cfg["faults"]["fault_fraction"]
    ns  = cfg["signal"]["noise_std"]

    mse_means = {m: [] for m in METHODS}
    mse_cis   = {m: [] for m in METHODS}
    for N in N_vals:
        tmp = dict(cfg["network"])
        tmp["N"] = N
        c2 = dict(cfg); c2["network"] = tmp
        rows = mc(c2, n, ff, ft, net_seed_var=True)
        for m in METHODS:
            mu, ci = mse_mean_ci(rows, m)
            mse_means[m].append(mu)
            mse_cis[m].append(ci)
        print(f"  N={N}  " + "  ".join(
            f"{m[:4]}={mse_means[m][-1]:.5f}" for m in METHODS))

    viz.plot_mse_vs_N(N_vals, mse_means, mse_cis)
    return dict(N_vals=N_vals, mse=mse_means, ci=mse_cis)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 10: Fault type comparison
# ─────────────────────────────────────────────────────────────────────────────

def exp10_fault_types(cfg):
    print("\n=== Exp 10: Fault Type Comparison ===")
    fracs, n = cfg["sweeps"]["fault_fractions"], cfg["sweeps"]["n_trials"]
    results  = {}
    for ft in cfg["sweeps"]["fault_types"]:
        mse_by_ff = []
        for ff in fracs:
            rows = mc(cfg, n, ff, ft)
            mse_by_ff.append(float(np.mean([r["mse"]["Proposed"] for r in rows])))
        results[ft] = mse_by_ff
        print(f"  {ft}: {[f'{v:.5f}' for v in mse_by_ff]}")

    # Also: drift with enhanced disagreement illustration
    g, P = build_net(cfg)
    flt  = cfg["faults"]
    fus  = cfg["fusion"]
    ns   = cfg["signal"]["noise_std"]
    s    = generate_signal(cfg["signal"]["T"], cfg["signal"]["type"], seed=0)
    rng  = np.random.default_rng(42)
    fm, fts = assign_faults_random(cfg["network"]["N"], 0.20, "drift", rng)
    Y    = generate_measurements(s, cfg["network"]["N"], ns, fm, fts,
                                  flt["fault_onset"], flt["drift_rate"],
                                  flt["burst_std"], seed=142)
    _, W, Ds, D_enh = fuse_proposed(Y, g["neighbors"], P,
                                     alpha=fus["alpha"], beta=fus["beta"],
                                     lag=fus["lag"], n_iter=fus["n_iter"])
    viz.plot_enhanced_vs_raw(Ds, D_enh, fm, flt["fault_onset"],
                              savename="fig4_enhanced_disagreement_drift.pdf")
    viz.plot_fault_type_comparison(fracs, results)
    return dict(fractions=fracs, fault_types=list(results.keys()), mse=results)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 11: Communication overhead vs MSE
# ─────────────────────────────────────────────────────────────────────────────

def exp11_comm_overhead(cfg):
    print("\n=== Exp 11: Communication Overhead vs MSE ===")
    n    = cfg["sweeps"]["n_trials"]
    fus  = cfg["fusion"]
    rows = mc(cfg, n, cfg["faults"]["fault_fraction"], cfg["faults"]["fault_type"])

    mse_scalar = {m: float(np.mean([r["mse"][m] for r in rows])) for m in METHODS}
    overhead   = communication_overhead(fus["n_iter"])

    # Map overhead keys to METHODS names
    oh_mapped = {
        "Average":         overhead["average"],
        "Trimmed Mean":    overhead["trimmed_mean"],
        "Local Median":    overhead["local_median"],
        "Plain Consensus": overhead["plain_consensus"],
        "Proposed":        overhead["proposed"],
    }
    print("  MSE:", {k: f"{v:.5f}" for k, v in mse_scalar.items()})
    print("  Overhead:", oh_mapped)
    viz.plot_comm_overhead(oh_mapped, mse_scalar)
    return dict(mse=mse_scalar, overhead=oh_mapped)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    if args.quick:
        print(">>> Quick mode: reduced trials & sweeps")
        cfg["sweeps"]["n_trials"]        = 5
        cfg["sweeps"]["fault_fractions"] = [0.10, 0.20, 0.30]
        cfg["sweeps"]["alphas"]          = [0.0, 0.5, 0.85, 0.99]
        cfg["sweeps"]["betas"]           = [1.0, 2.0, 3.0]
        cfg["sweeps"]["lags"]            = [5, 10, 20]
        cfg["sweeps"]["n_iter_vals"]     = [5, 20, 40]
        cfg["sweeps"]["N_values"]        = [10, 20, 30]
        cfg["heterogeneous_noise"]["noisy_fractions"] = [0.0, 0.20, 0.40]
        cfg["network"]["N"]              = 20
        cfg["signal"]["T"]               = 150
        cfg["fusion"]["n_iter"]          = 20

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    summary = {}
    r1  = exp1_illustration(cfg)
    summary["exp2_enhanced"]   = exp2_enhanced_disagreement(cfg)
    summary["exp3_mse_ff"]     = exp3_mse_vs_fault_fraction(cfg)
    summary["exp4_detectors"]  = exp4_detector_comparison(cfg)
    summary["exp5_clustered"]   = exp5_clustered_faults(cfg)
    summary["exp6_alpha"]      = exp6_alpha_sweep(cfg)
    summary["exp7_heatmap"]    = exp7_sensitivity_heatmap(cfg)
    summary["exp8_convergence"]= exp8_consensus_convergence(cfg)
    summary["exp9_N"]          = exp9_network_size(cfg)
    summary["exp10_fault_type"]= exp10_fault_types(cfg)
    summary["exp11_overhead"]  = exp11_comm_overhead(cfg)

    with open("results/summary.json", "w") as f:
        json.dump(clean_for_json(summary), f, indent=2, default=float)

    print("\n" + "="*60)
    print("✓ All 11 experiments complete")
    print("✓ 13 figures → results/figures/")
    print("✓ Numeric summary → results/summary.json")
    print("="*60)


if __name__ == "__main__":
    main()
