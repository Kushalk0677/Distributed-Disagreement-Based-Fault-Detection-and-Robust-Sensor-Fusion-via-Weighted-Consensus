#!/usr/bin/env python3
"""
run_real_data.py
----------------
Runs the proposed fault-detection and fusion pipeline on four real datasets.

Usage
-----
  python experiments/run_real_data.py                      # all available
  python experiments/run_real_data.py --dataset berkeley
  python experiments/run_real_data.py --dataset airquality
  python experiments/run_real_data.py --dataset smartcity
  python experiments/run_real_data.py --dataset maintenance
  python experiments/run_real_data.py --list               # show what's available

Place raw data files in:
  data/raw/BerkeleyLab.txt
  data/raw/AirQualityUCI.csv
  data/raw/smart_city_sensor_data.csv
  data/raw/sensor_maintenance_data.csv
"""

import sys, argparse, json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets  import load_dataset, DATASET_REGISTRY
from src.network   import (build_random_geometric_graph,
                            build_metropolis_weights,
                            assign_faults_random)
from src.fusion    import (fuse_average, fuse_trimmed_mean,
                            fuse_local_median, fuse_consensus_plain,
                            fuse_proposed)
from src.detection import DisagreementDetector, CUSUMDetector, EWMADetector
from src.metrics   import mse, rmse, detection_metrics
import src.visualize as viz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "lines.linewidth": 1.2, "axes.grid": True,
    "grid.alpha": 0.3,
})

FIGDIR = Path("results/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

FUSION_PARAMS = dict(alpha=0.85, beta=2.0, lag=25, n_iter=40, trim=0.10)
DET_PARAMS    = dict(k=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Network builder from dataset positions
# ─────────────────────────────────────────────────────────────────────────────

def build_net_from_data(positions: np.ndarray, radius: float = 0.32,
                         seed: int = 42):
    """Build RGG + Metropolis weights from dataset's sensor positions."""
    N   = len(positions)
    # Override positions in graph builder
    import src.network as nw
    g   = nw.build_random_geometric_graph(N, radius, seed=seed)
    g["positions"] = positions          # use real positions
    # Recompute adjacency from real positions
    adj       = np.zeros((N, N), dtype=bool)
    neighbors = [[] for _ in range(N)]
    r         = radius
    # Increase radius until connected
    for _ in range(15):
        adj       = np.zeros((N, N), dtype=bool)
        neighbors = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(positions[i] - positions[j]) <= r:
                    adj[i, j] = adj[j, i] = True
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        if nw._is_connected(neighbors, N):
            break
        r *= 1.15
    g["adj_matrix"] = adj
    g["neighbors"]  = neighbors
    g["degrees"]    = np.array([len(nb) for nb in neighbors])
    P = nw.build_metropolis_weights(g)
    return g, P


# ─────────────────────────────────────────────────────────────────────────────
# Reference signal estimation (median across sensors over time)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_reference(Y: np.ndarray) -> np.ndarray:
    """
    For real data there is no ground-truth signal. We use the temporal
    median across all sensors as a best-available reference.
    """
    return np.median(Y, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Single dataset experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset_experiment(name: str, data_dir: str = "data/raw",
                            dataset_kwargs: dict = None) -> dict:
    """
    Full pipeline on one dataset:
      1. Load & parse
      2. Build sensor network from positions
      3. Run all fusion methods
      4. Run all detectors
      5. Save figures
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    print(f"\n{'='*60}")
    print(f"  Dataset: {name.upper()}")
    print(f"{'='*60}")

    data = load_dataset(name, data_dir=data_dir, **dataset_kwargs)

    Y         = data["Y"]          # (N, T) normalised
    positions = data["positions"]  # (N, 2)
    N, T      = Y.shape
    fault_mask  = data["fault_mask"]
    has_labels  = fault_mask is not None and fault_mask.any()

    print(f"  Shape: {N} sensors × {T} time steps")
    if has_labels:
        print(f"  Fault fraction: {fault_mask.mean():.1%}"
              f"  ({fault_mask.sum()} faulty sensors)")

    # Network
    g, P = build_net_from_data(positions)
    mean_deg = g["degrees"].mean()
    print(f"  Network: mean degree = {mean_deg:.1f}, "
          f"radius = {g['radius']:.3f}")

    # ── Reference signal (median baseline for MSE) ──────────────────────────
    s_ref = estimate_reference(Y)

    # ── Fusion ───────────────────────────────────────────────────────────────
    fp = FUSION_PARAMS
    s_avg   = fuse_average(Y)
    s_tri   = fuse_trimmed_mean(Y, trim=fp["trim"])
    s_lmed  = fuse_local_median(Y, g["neighbors"])
    s_pcon  = fuse_consensus_plain(Y, P, n_iter=fp["n_iter"])
    s_prop, W, Ds, D_enh = fuse_proposed(
        Y, g["neighbors"], P,
        alpha=fp["alpha"], beta=fp["beta"],
        lag=fp["lag"], n_iter=fp["n_iter"],
    )

    estimates = {
        "Average":         s_avg,
        "Trimmed Mean":    s_tri,
        "Local Median":    s_lmed,
        "Plain Consensus": s_pcon,
        "Proposed":        s_prop,
    }
    mse_scores = {k: mse(s_ref, v) for k, v in estimates.items()}

    print("\n  MSE vs. reference (lower = better):")
    for k, v in mse_scores.items():
        print(f"    {k:<20} {v:.6f}")

    # ── Detection ────────────────────────────────────────────────────────────
    det_results = {}
    fault_onset = 0   # real data: evaluate over full window

    dd = DisagreementDetector(alpha=fp["alpha"], beta=fp["beta"],
                               lag=fp["lag"], k=DET_PARAMS["k"])
    cd = CUSUMDetector()
    ed = EWMADetector()

    pred_dd, _, _ = dd.fit_predict(Y, g["neighbors"], fault_onset)
    pred_cd, _, _ = cd.fit_predict(Y, g["neighbors"], fault_onset)
    pred_ed, _, _ = ed.fit_predict(Y, g["neighbors"], fault_onset)

    print(f"\n  Flagged as faulty:")
    print(f"    Proposed:  {pred_dd.sum()}/{N} sensors")
    print(f"    CUSUM:     {pred_cd.sum()}/{N} sensors")
    print(f"    EWMA:      {pred_ed.sum()}/{N} sensors")

    if has_labels:
        for det_name, pred in [("Proposed", pred_dd),
                                ("CUSUM",    pred_cd),
                                ("EWMA",     pred_ed)]:
            dm = detection_metrics(fault_mask, pred)
            det_results[det_name] = dm
            print(f"    {det_name:<12} FDR={dm['FDR']:.2f}  "
                  f"FAR={dm['FAR']:.2f}  F1={dm['F1']:.2f}")

    # ── Figures ──────────────────────────────────────────────────────────────
    slug = name.lower().replace(" ", "_")
    t    = np.arange(T)

    # Fig A: Topology with detected faults highlighted
    _plot_real_topology(g, pred_dd, fault_mask,
                         title=f"{data['meta']['dataset']} — Network",
                         savename=f"real_{slug}_topology.pdf")

    # Fig B: Estimation comparison
    _plot_real_estimation(t, s_ref, estimates, data["channel"],
                           data["meta"]["dataset"],
                           savename=f"real_{slug}_estimation.pdf")

    # Fig C: Weight heatmap (sensors × time)
    _plot_weight_heatmap(W, fault_mask,
                          title=f"{data['meta']['dataset']} — Reliability Weights",
                          savename=f"real_{slug}_weights.pdf")

    # Fig D: Disagreement over time (enhanced)
    _plot_disagreement_traces(D_enh, pred_dd,
                               title=f"{data['meta']['dataset']} — Enhanced Disagreement",
                               savename=f"real_{slug}_disagreement.pdf")

    # Fig E: MSE bar chart
    _plot_mse_bar(mse_scores,
                   title=f"{data['meta']['dataset']} — MSE vs. Reference",
                   savename=f"real_{slug}_mse_bar.pdf")

    result = dict(
        dataset=name,
        N=N, T=T,
        channel=data["channel"],
        mse=mse_scores,
        n_flagged=dict(Proposed=int(pred_dd.sum()),
                       CUSUM=int(pred_cd.sum()),
                       EWMA=int(pred_ed.sum())),
        detection=det_results if has_labels else None,
        has_labels=has_labels,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers (real-data specific)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"Average": "#ff7f0e", "Trimmed Mean": "#e377c2",
          "Local Median": "#2ca02c", "Plain Consensus": "#9467bd",
          "Proposed": "#1f77b4"}


def _plot_real_topology(g, pred_mask, gt_mask, title, savename):
    pos, adj = g["positions"], g["adj_matrix"]
    N = len(pos)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                        "k-", alpha=0.12, lw=0.5, zorder=1)
    colors = []
    for i in range(N):
        if gt_mask is not None and gt_mask[i]:
            colors.append("#d62728")       # known faulty
        elif pred_mask[i]:
            colors.append("#ff7f0e")       # detected
        else:
            colors.append("#1f77b4")       # healthy
    ax.scatter(pos[:,0], pos[:,1], c=colors, s=45, zorder=3,
               edgecolors="white", lw=0.3)
    handles = [mpatches.Patch(color="#1f77b4", label="Healthy"),
               mpatches.Patch(color="#ff7f0e", label="Flagged")]
    if gt_mask is not None and gt_mask.any():
        handles.append(mpatches.Patch(color="#d62728", label="Ground-truth fault"))
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_title(title, fontsize=8); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(FIGDIR / savename); plt.close(fig)
    print(f"  Saved {FIGDIR / savename}")


def _plot_real_estimation(t, s_ref, estimates, channel, dataset, savename):
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.5), sharex=True)
    ax0, ax1 = axes

    # Top: all methods vs reference
    ax0.plot(t, s_ref, "k--", lw=1.5, label="Reference (median)")
    for label, s_hat in estimates.items():
        ax0.plot(t, s_hat, color=COLORS.get(label, "gray"), alpha=0.75,
                 lw=0.9, label=label)
    ax0.set_ylabel(f"{channel} (normalised)")
    ax0.set_title(f"{dataset} — Estimation Comparison")
    ax0.legend(fontsize=6, ncol=3)

    # Bottom: residuals of proposed vs average
    ax1.fill_between(t, s_ref - estimates["Average"],
                     alpha=0.3, color="#ff7f0e", label="Average residual")
    ax1.fill_between(t, s_ref - estimates["Proposed"],
                     alpha=0.3, color="#1f77b4", label="Proposed residual")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("Time step"); ax1.set_ylabel("Residual")
    ax1.set_title("Residuals vs. Reference")
    ax1.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGDIR / savename); plt.close(fig)
    print(f"  Saved {FIGDIR / savename}")


def _plot_weight_heatmap(W, fault_mask, title, savename):
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    im = ax.imshow(W, aspect="auto", origin="upper",
                   cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Reliability weight $w_i$")
    if fault_mask is not None:
        for i, f in enumerate(fault_mask):
            if f:
                ax.axhline(i, color="red", lw=0.4, alpha=0.6)
    ax.set_xlabel("Time step"); ax.set_ylabel("Sensor index")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGDIR / savename); plt.close(fig)
    print(f"  Saved {FIGDIR / savename}")


def _plot_disagreement_traces(D_enh, pred_mask, title, savename):
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for i in range(D_enh.shape[0]):
        c = "#d62728" if pred_mask[i] else "#1f77b4"
        ax.plot(D_enh[i], color=c, alpha=0.8 if pred_mask[i] else 0.15,
                lw=1.2 if pred_mask[i] else 0.4)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Enhanced disagreement $D_i^+$")
    ax.set_title(title)
    ax.legend(handles=[
        mpatches.Patch(color="#1f77b4", label="Normal"),
        mpatches.Patch(color="#d62728", label="Flagged as faulty"),
    ], fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / savename); plt.close(fig)
    print(f"  Saved {FIGDIR / savename}")


def _plot_mse_bar(mse_scores, title, savename):
    labels = list(mse_scores.keys())
    values = list(mse_scores.values())
    colors = [COLORS.get(l, "gray") for l in labels]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("MSE vs. reference")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / savename); plt.close(fig)
    print(f"  Saved {FIGDIR / savename}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset summary figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_dataset_summary(all_results: dict, savename="real_summary.pdf"):
    """Bar chart comparing proposed MSE reduction vs. average across datasets."""
    datasets = [r["dataset"] for r in all_results.values()]
    avg_mse  = [r["mse"]["Average"]  for r in all_results.values()]
    prop_mse = [r["mse"]["Proposed"] for r in all_results.values()]
    lmed_mse = [r["mse"]["Local Median"] for r in all_results.values()]

    x     = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(x - width, avg_mse,  width, label="Average",       color="#ff7f0e")
    ax.bar(x,         lmed_mse, width, label="Local Median",  color="#2ca02c")
    ax.bar(x + width, prop_mse, width, label="Proposed",      color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylabel("MSE vs. reference")
    ax.set_title("Real-World Dataset Comparison — All Methods")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / savename); plt.close(fig)
    print(f"\n  Saved {FIGDIR / savename}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific kwargs
# ─────────────────────────────────────────────────────────────────────────────

DATASET_KWARGS = {
    "berkeley":    dict(channel="temperature", max_T=2000),
    "airquality":  dict(channel="oxide_sensors", max_T=2000),
    "smartcity":   dict(sensor_type_filter="all", max_T=2000),
    "maintenance": dict(channel="temperature", max_T=500),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="all",
                   help="Dataset to run: berkeley|airquality|smartcity|maintenance|all")
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--list",     action="store_true",
                   help="List available datasets and expected file names")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable datasets and expected raw file names:")
        for name, (_, fname) in DATASET_REGISTRY.items():
            fpath = Path(args.data_dir) / fname
            status = "✓ found" if fpath.exists() else "✗ missing"
            print(f"  {name:<15}  {fname:<40}  {status}")
        return

    # Decide which datasets to run
    if args.dataset == "all":
        to_run = list(DATASET_REGISTRY.keys())
    else:
        to_run = [args.dataset]

    # Skip datasets whose files are not present
    available = []
    for name in to_run:
        _, fname = DATASET_REGISTRY[name]
        fpath = Path(args.data_dir) / fname
        if fpath.exists():
            available.append(name)
        else:
            print(f"  [skip] {name}: file not found at {fpath}")

    if not available:
        print("\nNo dataset files found.")
        print(f"Place files in: {Path(args.data_dir).resolve()}")
        print("Run with --list to see expected filenames.")
        return

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    all_results = {}
    for name in available:
        try:
            result = run_dataset_experiment(
                name,
                data_dir=args.data_dir,
                dataset_kwargs=DATASET_KWARGS.get(name, {}),
            )
            all_results[name] = result
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback; traceback.print_exc()

    if len(all_results) > 1:
        plot_cross_dataset_summary(all_results)

    # Save JSON summary
    def clean(obj):
        if isinstance(obj, dict):  return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [clean(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer, np.bool_)): return float(obj)
        return obj

    out = Path("results/real_data_summary.json")
    with open(out, "w") as f:
        json.dump(clean(all_results), f, indent=2, default=float)
    print(f"\n✓ Summary saved to {out}")
    print(f"✓ Figures saved to results/figures/")


import json
if __name__ == "__main__":
    main()
