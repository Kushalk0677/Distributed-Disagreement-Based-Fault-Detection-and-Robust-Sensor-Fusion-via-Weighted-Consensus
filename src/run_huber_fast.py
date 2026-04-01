#!/usr/bin/env python3
"""
run_huber_fast.py — optimized single-pass version
Runs all methods in a single trial loop to avoid redundant computation.
"""

import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network import (build_random_geometric_graph,
                          assign_faults_random, assign_faults_clustered,
                          build_metropolis_weights)
from src.signal  import generate_signal, generate_measurements
from src.fusion  import (fuse_proposed, fuse_distributed_kf, fuse_huber_dkf)
from src.metrics import mse, ci95

CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
OUT_DIR  = Path(__file__).resolve().parent.parent / "results"
FIG_DIR  = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

N_TRIALS      = 50
FAULT_FRACS   = [0.05, 0.10, 0.20, 0.30, 0.40]
HUBER_C_SWEEP = [0.5, 1.0, 1.5, 2.0, 3.0]
N             = cfg["network"]["N"]
RADIUS        = cfg["network"]["radius"]
SEED          = cfg["network"]["seed"]
T             = cfg["signal"]["T"]
NOISE_STD     = cfg["signal"]["noise_std"]
FAULT_ONSET   = cfg["faults"]["fault_onset"]
DRIFT_RATE    = cfg["faults"]["drift_rate"]
BURST_STD     = cfg["faults"]["burst_std"]
N_ITER        = cfg["fusion"]["n_iter"]
ALPHA         = cfg["fusion"]["alpha"]
BETA          = cfg["fusion"]["beta"]
LAG           = cfg["fusion"]["lag"]
Q_KF, R_KF   = 1e-3, 0.0025

print("Building network...")
g = build_random_geometric_graph(N, RADIUS, seed=SEED)
P = build_metropolis_weights(g)
# Precompute P^n_iter for KF methods
P_pow = np.linalg.matrix_power(P, N_ITER)

conditions = [
    ("stuck", False, "Stuck / Random"),
    ("stuck", True,  "Stuck / Clustered"),
    ("drift", False, "Drift / Random"),
]

all_results = {}

for fault_type, clustered, label in conditions:
    print(f"\n{'='*55}\nCondition: {label}\n{'='*55}")
    cond = {}

    for ff in FAULT_FRACS:
        print(f"  ff={ff:.0%} ...", end=" ", flush=True)

        prop_vals = []
        dkf_vals  = []
        huber_by_c = {c: [] for c in HUBER_C_SWEEP}

        for trial in range(N_TRIALS):
            s   = generate_signal(T, signal_type="sinusoid", seed=trial)
            rng = np.random.default_rng(trial * 7)

            if clustered:
                fm, ft = assign_faults_clustered(
                    g["positions"], ff, fault_type, rng)
            else:
                fm, ft = assign_faults_random(N, ff, fault_type, rng)

            Y = generate_measurements(
                s, N, NOISE_STD, fault_mask=fm, fault_types=ft,
                fault_onset=FAULT_ONSET, drift_rate=DRIFT_RATE,
                burst_std=BURST_STD, seed=trial * 7 + 100)

            # Proposed
            sp, _, _, _ = fuse_proposed(
                Y, g["neighbors"], P,
                alpha=ALPHA, beta=BETA, lag=LAG, n_iter=N_ITER)
            prop_vals.append(mse(s, sp))

            # Plain DKF (reuse P_pow)
            x_hat = np.zeros(N)
            p_var = np.ones(N) * R_KF
            s_dkf = np.zeros(T)
            for t in range(T):
                x_pred = x_hat.copy()
                p_pred = p_var + Q_KF
                K_k    = p_pred / (p_pred + R_KF)
                x_hat  = x_pred + K_k * (Y[:, t] - x_pred)
                p_var  = (1.0 - K_k) * p_pred
                x_hat  = P_pow @ x_hat
                s_dkf[t] = x_hat.mean()
            dkf_vals.append(mse(s, s_dkf))

            # Huber-DKF for each c
            for c in HUBER_C_SWEEP:
                x_hat2 = np.zeros(N)
                p_var2 = np.ones(N) * R_KF
                s_hub  = np.zeros(T)
                for t in range(T):
                    x_pred2 = x_hat2.copy()
                    p_pred2 = p_var2 + Q_KF
                    K_k2    = p_pred2 / (p_pred2 + R_KF)
                    innov   = Y[:, t] - x_pred2
                    clipped = np.where(np.abs(innov) <= c, innov, c * np.sign(innov))
                    x_hat2  = x_pred2 + K_k2 * clipped
                    p_var2  = (1.0 - K_k2) * p_pred2
                    x_hat2  = P_pow @ x_hat2
                    s_hub[t] = x_hat2.mean()
                huber_by_c[c].append(mse(s, s_hub))

        # Find best c
        best_c    = min(HUBER_C_SWEEP, key=lambda c: np.mean(huber_by_c[c]))
        best_vals = huber_by_c[best_c]

        pm, pci = ci95(prop_vals)
        dm, dci = ci95(dkf_vals)
        hm, hci = ci95(best_vals)

        cond[ff] = dict(
            proposed_mean=pm, proposed_ci=pci,
            dkf_mean=dm,      dkf_ci=dci,
            huber_mean=hm,    huber_ci=hci,
            best_c=best_c,
            ratio_huber_proposed=hm/pm if pm>0 else None,
            ratio_dkf_proposed=dm/pm   if pm>0 else None,
        )

        # Also store all c results for appendix
        cond[ff]["huber_by_c"] = {
            c: {"mean": float(np.mean(v)), "ci": float(ci95(v)[1])}
            for c, v in huber_by_c.items()
        }

        print(f"Proposed={pm*1e3:.3f}  Huber(c={best_c})={hm*1e3:.3f}  "
              f"DKF={dm*1e3:.3f}  ratio={hm/pm:.2f}x")

    all_results[label] = cond

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / "huber_comparison.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved → {out_path}")

# ── Print LaTeX-ready table ───────────────────────────────────────────────────
print("\n" + "="*80)
print("LaTeX TABLE DATA (MSE ×10⁻³, ±1.96 SE)")
print("="*80)
print(f"{'Fault/Placement':<22} {'FF':>5}  {'Proposed':>14}  {'Huber-DKF':>14}  {'Plain DKF':>14}  {'c*':>4}")
print("-"*80)
for label, cond in all_results.items():
    for ff, r in sorted(cond.items()):
        if isinstance(ff, float):
            print(f"{label:<22} {ff:>5.0%}  "
                  f"{r['proposed_mean']*1e3:>7.3f}±{r['proposed_ci']*1e3:.3f}  "
                  f"{r['huber_mean']*1e3:>7.3f}±{r['huber_ci']*1e3:.3f}  "
                  f"{r['dkf_mean']*1e3:>7.3f}±{r['dkf_ci']*1e3:.3f}  "
                  f"{r['best_c']:>4}")

# ── Figure: MSE vs fault fraction ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
col = {"Proposed": "#2196F3", "Huber-DKF": "#FF9800", "Plain DKF": "#9E9E9E"}

for ax, (label, cond) in zip(axes, all_results.items()):
    ffs  = sorted(k for k in cond if isinstance(k, float))
    pcts = [f*100 for f in ffs]

    pm = [cond[f]["proposed_mean"]*1e3 for f in ffs]
    pc = [cond[f]["proposed_ci"]*1e3   for f in ffs]
    hm = [cond[f]["huber_mean"]*1e3    for f in ffs]
    hc = [cond[f]["huber_ci"]*1e3      for f in ffs]
    dm = [cond[f]["dkf_mean"]*1e3      for f in ffs]
    dc = [cond[f]["dkf_ci"]*1e3        for f in ffs]

    ax.errorbar(pcts, pm, yerr=pc, marker="o",
                color=col["Proposed"],  label="Proposed",  lw=2,   capsize=4)
    ax.errorbar(pcts, hm, yerr=hc, marker="s",
                color=col["Huber-DKF"], label="Huber-DKF", lw=2,   capsize=4, ls="--")
    ax.errorbar(pcts, dm, yerr=dc, marker="^",
                color=col["Plain DKF"], label="Plain DKF", lw=1.5, capsize=4, ls=":")

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Fault fraction (%)", fontsize=10)
    ax.set_ylabel("MSE (×10⁻³)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Proposed vs Fault-Tolerant DKF (Huber, best c over [0.5–3.0])",
             fontsize=11, y=1.01)
plt.tight_layout()
p1 = FIG_DIR / "fig_huber_mse_comparison.pdf"
plt.savefig(p1, bbox_inches="tight")
plt.savefig(str(p1).replace(".pdf",".png"), bbox_inches="tight", dpi=150)
print(f"Figure → {p1}")
plt.close()

# ── Figure: ratio bar chart at 20% ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
labels_short = ["Stuck\nRandom", "Stuck\nClustered", "Drift\nRandom"]
rh = [all_results[l][0.20]["ratio_huber_proposed"] for l in all_results]
rd = [all_results[l][0.20]["ratio_dkf_proposed"]   for l in all_results]

x = np.arange(3)
w = 0.35
b1 = ax.bar(x-w/2, rh, w, color=col["Huber-DKF"], label="Huber-DKF / Proposed", alpha=0.85)
b2 = ax.bar(x+w/2, rd, w, color=col["Plain DKF"], label="Plain DKF / Proposed",  alpha=0.85)
ax.axhline(1.0, color="black", lw=1.5, ls="--", label="Proposed (1.0×)")
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=11)
ax.set_ylabel("MSE ratio (method / Proposed)", fontsize=11)
ax.set_title("Relative MSE at 20% Fault Fraction (lower = Proposed wins)", fontsize=11)
ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)
for bar in list(b1)+list(b2):
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.04, f"{h:.2f}×",
            ha="center", va="bottom", fontsize=9)
plt.tight_layout()
p2 = FIG_DIR / "fig_huber_ratio_20pct.pdf"
plt.savefig(p2, bbox_inches="tight")
plt.savefig(str(p2).replace(".pdf",".png"), bbox_inches="tight", dpi=150)
print(f"Figure → {p2}")
plt.close()

print("\nDone.")
