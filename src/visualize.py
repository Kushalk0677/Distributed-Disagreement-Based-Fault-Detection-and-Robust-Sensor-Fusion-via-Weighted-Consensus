"""
visualize.py — publication-quality IEEE Sensors Letters figures

All figures saved as PDFs in results/figures/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

FIGDIR = Path("results/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":      150,
    "lines.linewidth":  1.2,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linewidth":  0.5,
})

# Consistent color / style palette for all figures
PALETTE = {
    "Average":          ("#d62728", "-o"),
    "Trimmed Mean":     ("#ff7f0e", "-s"),
    "Local Median":     ("#2ca02c", "-^"),
    "Plain Consensus":  ("#9467bd", "-D"),
    "Proposed":         ("#1f77b4", "-*"),
    "Ground truth":     ("black",   "--"),
    "CUSUM":            ("#e377c2", "-o"),
    "EWMA":             ("#8c564b", "-s"),
    "Disagreement":     ("#1f77b4", "-^"),
}


def _save(fig, name):
    p = FIGDIR / name
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")


def _method_line(ax, x, y, label, yerr=None, **kw):
    color, marker = PALETTE.get(label, ("gray", "-o"))
    style = marker[0]           # line style
    mkr   = marker[1:]          # marker
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt=style + mkr,
                    color=color, label=label, capsize=3, **kw)
    else:
        ax.plot(x, y, style + mkr, color=color, label=label, **kw)


# ── Fig 1: Topology ──────────────────────────────────────────────────────────
def plot_topology(graph, fault_mask, hi_noise_mask=None,
                  savename="fig1_topology.pdf"):
    pos, adj = graph["positions"], graph["adj_matrix"]
    N = len(pos)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                        "k-", alpha=0.12, lw=0.5, zorder=1)

    colors = []
    for i in range(N):
        if fault_mask[i]:
            colors.append("#d62728")
        elif hi_noise_mask is not None and hi_noise_mask[i]:
            colors.append("#ff7f0e")
        else:
            colors.append("#1f77b4")
    ax.scatter(pos[:,0], pos[:,1], c=colors, s=45, zorder=3, edgecolors="white", lw=0.3)

    handles = [mpatches.Patch(color="#1f77b4", label="Healthy")]
    if hi_noise_mask is not None:
        handles.append(mpatches.Patch(color="#ff7f0e", label="High-noise"))
    handles.append(mpatches.Patch(color="#d62728", label="Faulty"))
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Sensor Network Topology")
    ax.set_aspect("equal")
    _save(fig, savename)


# ── Fig 2: Estimation traces ─────────────────────────────────────────────────
def plot_estimation(t, s_true, estimates, fault_onset,
                    savename="fig2_estimation.pdf"):
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(t, s_true, color="black", ls="--", lw=1.5, label="Ground truth")
    ax.axvline(fault_onset, color="gray", ls=":", lw=1.0, label="Fault onset")
    for label, s_hat in estimates.items():
        color, marker = PALETTE.get(label, ("gray", "-"))
        ax.plot(t, s_hat, color=color, ls=marker[0], label=label, alpha=0.85)
    ax.set_xlabel("Time step"); ax.set_ylabel("Signal value")
    ax.set_title("Estimation Comparison"); ax.legend(ncol=2, fontsize=7)
    _save(fig, savename)


# ── Fig 3: Per-sensor weights ────────────────────────────────────────────────
def plot_weights(W, fault_mask, fault_onset, savename="fig3_weights.pdf"):
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    for i in range(len(fault_mask)):
        c = "#d62728" if fault_mask[i] else "#1f77b4"
        ax.plot(W[i], color=c, alpha=0.8 if fault_mask[i] else 0.18,
                lw=1.2 if fault_mask[i] else 0.4)
    ax.axvline(fault_onset, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Time step"); ax.set_ylabel("Reliability weight $w_i$")
    ax.set_title("Per-Sensor Reliability Weights"); ax.set_ylim(0, 1.05)
    ax.legend(handles=[
        mpatches.Patch(color="#1f77b4", label="Healthy"),
        mpatches.Patch(color="#d62728", label="Faulty"),
    ])
    _save(fig, savename)


# ── Fig 4: Enhanced vs raw disagreement (drift) ──────────────────────────────
def plot_enhanced_vs_raw(Ds, D_enh, fault_mask, fault_onset,
                          savename="fig4_enhanced_disagreement.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for ax, D, title in zip(axes,
                             [Ds, D_enh],
                             ["Smoothed $D_i$", "Enhanced $D_i^+$ (trend augmented)"]):
        for i in range(D.shape[0]):
            c = "#d62728" if fault_mask[i] else "#1f77b4"
            ax.plot(D[i], color=c, alpha=0.8 if fault_mask[i] else 0.15,
                    lw=1.2 if fault_mask[i] else 0.4)
        ax.axvline(fault_onset, color="gray", ls="--", lw=1.0)
        ax.set_title(title); ax.set_xlabel("Time step"); ax.set_ylabel("Score")
    axes[0].legend(handles=[
        mpatches.Patch(color="#1f77b4", label="Healthy"),
        mpatches.Patch(color="#d62728", label="Faulty (drift)"),
    ], fontsize=7)
    _save(fig, savename)


# ── Fig 5: MSE vs fault fraction (all methods) ───────────────────────────────
def plot_mse_vs_fault_fraction(fractions, mse_results, ci_results=None,
                                savename="fig5_mse_vs_fault_fraction.pdf"):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for label, vals in mse_results.items():
        yerr = ci_results.get(label) if ci_results else None
        _method_line(ax, fractions, vals, label, yerr=yerr, markersize=5)
    ax.set_xlabel("Fault fraction")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs. Fault Fraction — All Methods")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=7)
    _save(fig, savename)


# ── Fig 6: FDR / FAR / F1 — detector comparison ──────────────────────────────
def plot_detector_comparison(fractions, det_results,
                              savename="fig6_detector_comparison.pdf"):
    """
    det_results = {
      'Proposed':  {'fdr': [...], 'far': [...], 'f1': [...]},
      'CUSUM':     {'fdr': [...], 'far': [...], 'f1': [...]},
      'EWMA':      {'fdr': [...], 'far': [...], 'f1': [...]},
    }
    """
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8), sharey=True)
    titles = ["FDR (Recall)", "FAR", "F1 Score"]
    keys   = ["fdr", "far", "f1"]
    for ax, title, key in zip(axes, titles, keys):
        for label, d in det_results.items():
            color, marker = PALETTE.get(label, ("gray", "-o"))
            ax.plot(fractions, d[key], marker[0] + marker[1:],
                    color=color, label=label, markersize=5)
        ax.set_title(title)
        ax.set_xlabel("Fault fraction")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_ylim(-0.05, 1.1)
    axes[0].set_ylabel("Rate")
    axes[0].legend(fontsize=7)
    _save(fig, savename)


# ── Fig 7: Heterogeneous noise — key result ───────────────────────────────────
def plot_heterogeneous_noise(noisy_fracs, mse_results, ci_results=None,
                              savename="fig7_heterogeneous_noise.pdf"):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for label, vals in mse_results.items():
        yerr = ci_results.get(label) if ci_results else None
        _method_line(ax, noisy_fracs, vals, label, yerr=yerr, markersize=5)
    ax.set_xlabel("Fraction of high-noise sensors")
    ax.set_ylabel("MSE")
    ax.set_title("Heterogeneous Noise: Proposed vs. Baselines\n"
                 "(No faults — proposed beats local median via noise-aware weighting)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=7)
    _save(fig, savename)


# ── Fig 8: MSE vs alpha sweep ─────────────────────────────────────────────────
def plot_mse_vs_alpha(alphas, mse_vals, savename="fig8_mse_vs_alpha.pdf"):
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(alphas, mse_vals, "o-", color="#1f77b4", markersize=5)
    best_a = alphas[np.argmin(mse_vals)]
    ax.axvline(best_a, color="red", ls="--", lw=1.0,
               label=f"Optimal α = {best_a:.2f}")
    ax.set_xlabel(r"Smoothing factor $\alpha$"); ax.set_ylabel("MSE")
    ax.set_title(r"MSE vs. Smoothing Factor $\alpha$"); ax.legend()
    _save(fig, savename)


# ── Fig 9: 2-D sensitivity heatmap (beta vs lag) ─────────────────────────────
def plot_sensitivity_heatmap(betas, lags, mse_grid,
                              savename="fig9_sensitivity_heatmap.pdf"):
    """
    mse_grid : (len(betas), len(lags)) array
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(mse_grid, aspect="auto", origin="lower",
                   cmap="RdYlGn_r",
                   extent=[lags[0]-0.5, lags[-1]+0.5,
                            betas[0]-0.25, betas[-1]+0.25])
    fig.colorbar(im, ax=ax, label="MSE")
    best = np.unravel_index(np.argmin(mse_grid), mse_grid.shape)
    ax.scatter(lags[best[1]], betas[best[0]], marker="*",
               s=150, c="white", zorder=5,
               label=f"Best: β={betas[best[0]]:.1f}, lag={lags[best[1]]}")
    ax.set_xlabel("Trend lag"); ax.set_ylabel(r"Trend weight $\beta$")
    ax.set_yticks(betas)
    ax.set_xticks(lags)
    ax.set_title(r"MSE Sensitivity: $\beta$ vs. Trend Lag")
    ax.legend(fontsize=7)
    _save(fig, savename)


# ── Fig 10: Consensus convergence ────────────────────────────────────────────
def plot_consensus_convergence(n_iter_vals, mse_proposed, mse_plain,
                                savename="fig10_consensus_convergence.pdf"):
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.semilogy(n_iter_vals, mse_proposed, "o-", color="#1f77b4",
                label="Proposed (weighted)", markersize=5)
    ax.semilogy(n_iter_vals, mse_plain, "s--", color="#9467bd",
                label="Plain consensus", markersize=5)
    ax.set_xlabel("Consensus iterations $K$")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("MSE vs. Consensus Iterations")
    ax.legend()
    _save(fig, savename)


# ── Fig 11: MSE vs network size ───────────────────────────────────────────────
def plot_mse_vs_N(N_vals, mse_results, ci_results=None,
                  savename="fig11_mse_vs_N.pdf"):
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for label, vals in mse_results.items():
        yerr = ci_results.get(label) if ci_results else None
        _method_line(ax, N_vals, vals, label, yerr=yerr, markersize=5)
    ax.set_xlabel("Number of sensors $N$"); ax.set_ylabel("MSE")
    ax.set_title("MSE vs. Network Size"); ax.legend(fontsize=7)
    _save(fig, savename)


# ── Fig 12: Fault type comparison ─────────────────────────────────────────────
def plot_fault_type_comparison(fractions, results,
                                savename="fig12_fault_type_comparison.pdf"):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    styles = ["-o", "-s", "-^", "-D"]
    for idx, (ft, vals) in enumerate(results.items()):
        ax.plot(fractions, vals, styles[idx], color=colors[idx],
                label=ft.capitalize(), markersize=5)
    ax.set_xlabel("Fault fraction"); ax.set_ylabel("MSE (Proposed)")
    ax.set_title("Proposed Method: MSE by Fault Type")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend()
    _save(fig, savename)


# ── Fig 13: Communication overhead vs MSE tradeoff ───────────────────────────
def plot_comm_overhead(overhead_vals, mse_results,
                        savename="fig13_comm_overhead.pdf"):
    """
    overhead_vals : dict {method: rounds_per_timestep}
    mse_results   : dict {method: scalar MSE}
    """
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for label in mse_results:
        color, _ = PALETTE.get(label, ("gray", "-o"))
        ax.scatter(overhead_vals[label], mse_results[label],
                   color=color, s=80, zorder=3, label=label)
        ax.annotate(label, (overhead_vals[label], mse_results[label]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Communication rounds per time step")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs. Communication Overhead")
    ax.legend(fontsize=7)
    _save(fig, savename)


# ── Fig 14: Clustered vs random fault placement ───────────────────────────────
def plot_clustered_vs_random(fractions, results,
                              savename="fig14_clustered_vs_random.pdf"):
    """
    results = {
      'Proposed (random)':     [...],
      'Local Median (random)': [...],
      'Proposed (clustered)':  [...],
      'Local Median (clustered)': [...],
    }
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    color_map = {
        "Proposed (random)":        ("#1f77b4", "-o"),
        "Local Median (random)":    ("#2ca02c", "-^"),
        "Proposed (clustered)":     ("#1f77b4", "--o"),
        "Local Median (clustered)": ("#2ca02c", "--^"),
    }
    for label, vals in results.items():
        color, mk = color_map.get(label, ("gray", "-o"))
        ax.plot(fractions, vals, mk, color=color, label=label, markersize=5)
    ax.set_xlabel("Fault fraction")
    ax.set_ylabel("MSE")
    ax.set_title("Proposed vs. Local Median:\nRandom vs. Clustered Fault Placement")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=7)
    _save(fig, savename)


# ── Fig 15: Clustered fault topology ─────────────────────────────────────────
def plot_clustered_topology(graph, fault_mask_rand, fault_mask_clust,
                             savename="fig15_clustered_topology.pdf"):
    pos = graph["positions"]
    adj = graph["adj_matrix"]
    N   = len(pos)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    titles = ["Random fault placement", "Clustered fault placement"]
    for ax, fmask, title in zip(axes, [fault_mask_rand, fault_mask_clust], titles):
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j]:
                    ax.plot([pos[i,0],pos[j,0]], [pos[i,1],pos[j,1]],
                            "k-", alpha=0.12, lw=0.5, zorder=1)
        colors = ["#d62728" if fmask[i] else "#1f77b4" for i in range(N)]
        ax.scatter(pos[:,0], pos[:,1], c=colors, s=45, zorder=3,
                   edgecolors="white", lw=0.3)
        ax.set_title(title); ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    axes[0].legend(handles=[mpatches.Patch(color="#1f77b4", label="Healthy"),
                              mpatches.Patch(color="#d62728", label="Faulty")], fontsize=7)
    _save(fig, savename)
