# Distributed Disagreement-Based Fault Detection and Robust Sensor Fusion via Weighted Consensus

Simulation code accompanying the IEEE Sensors Letters submission:

> **"Distributed Disagreement-Based Fault Detection and Robust Sensor Fusion via Weighted Consensus"**  
> Kushal Khemani and Sujal Kosta

---

## Overview

A lightweight fully distributed fault detection and sensor fusion framework for sensor networks. Each sensor computes a **trend-augmented disagreement score** against its neighbors' median measurement and converts it to a reliability weight used in iterative weighted consensus. No centralized controller, prior noise statistics, or explicit fault models are required.

### Key hyperparameter defaults (aligned with paper)

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Temporal smoothing | α | 0.85 | Flat MSE for α ∈ [0.7, 0.95] |
| Trend weight | β | 2.0 | Monotone MSE improvement with β |
| Lookback window | L | **25** | ≈ ½ × fault onset time (t₀=50) |
| Consensus iterations | K | 80 | Sufficient for N ≤ 100 |
| Detection threshold | k | **1.5** | FAR ≤ Φ(−1.5) ≈ 0.067 (theoretical); empirical FAR ≤ 0.007 |

> ⚠️ **Important:** `k_threshold = 1.5` is the theoretically grounded value (Eq. 10 in the paper).
> Do **not** use k < 1.0 — this collapses the FAR theoretical bound to Φ(−k) > 0.15, making the
> guarantee vacuous even if empirical FAR remains low in practice.

---

## Repository Structure

```
repo/
├── src/
│   ├── network.py            # Random geometric graph topology builder
│   ├── signal.py             # Ground-truth signal + fault injection models
│   ├── fusion.py             # All fusion algorithms (average, median, proposed, Dist. KF)
│   ├── detection.py          # DisagreementDetector, CUSUM, EWMA
│   ├── metrics.py            # MSE, RMSE, FDR, FAR, F1, CI evaluation
│   ├── datasets.py           # Real-dataset loaders
│   └── visualize.py          # All paper figures
│
├── experiments/
│   ├── run_experiments.py    # Main MC experiment runner (12 experiments)
│   └── run_real_data.py      # Real-dataset validation runner
│
├── configs/
│   └── default.yaml          # All hyperparameters — edit here, not in code
│
├── results/
│   ├── figures/              # PDF figures (generated on run)
│   ├── summary.json          # Simulation numeric results
│   └── real_data_summary.json
│
├── data/raw/                 # Place dataset files here (see README_DATA.md)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Kushalk0677/Distributed-Disagreement-Based-Fault-Detection-and-Robust-Sensor-Fusion.git
cd repo

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running Experiments

### Full simulation (~15–30 min, 50 trials)
```bash
python experiments/run_experiments.py
```

### Quick mode for testing (~1–2 min)
```bash
python experiments/run_experiments.py --quick
```

### Real-dataset validation
```bash
python experiments/run_real_data.py --list          # check which datasets are present
python experiments/run_real_data.py                 # run all available
python experiments/run_real_data.py --dataset berkeley
python experiments/run_real_data.py --dataset airquality    # strongest result (7.7× over Average)
python experiments/run_real_data.py --dataset maintenance   # has ground-truth fault labels
```

> **Note:** The SmartCity dataset yields N=2 sensors after filtering — too small for meaningful
> consensus evaluation and excluded from the paper.

---

## Fault Models

| Type | Description | Detection difficulty |
|------|-------------|----------------------|
| `stuck` | Freezes at value at fault onset | Easy — large instantaneous disagreement |
| `drift` | Slowly increasing bias | Hard — requires trend term (β > 0, L = 25) |
| `malicious` | Uniform[smin−δ, smax+δ] | Easy — large disagreement |
| `noise_burst` | σ² → σ²_burst ≫ σ² | Easy — large disagreement |

The trend term (Eq. 4) provides ~29% MSE reduction for drift faults vs. β=0 at fault fractions 5–30%.

---

## Method (paper equations)

```
1. Instantaneous disagreement:   d_i(t)  = |y_i(t) − median_{j∈Nᵢ} y_j(t)|
2. Temporal smoothing:           D_i(t)  = α·D_i(t−1) + (1−α)·d_i(t)
3. Trend-augmented statistic:    D⁺_i(t) = D_i(t) + β·max{0, D_i(t) − D_i(t−L)}
4. Reliability weight:           w_i(t)  = 1 / (1 + D⁺_i(t))
5. Weighted consensus (K iters): x_i^(k+1) = Σ_j P_ij·w_j·x_j^(k) / Σ_j P_ij·w_j
6. Global estimate:              ŝ(t)    = (1/N) Σ_i x_i^(K)
7. Fault detection:              τ = μ_w̄ − k·σ_w̄;   f̂_i = 1[w̄_i < τ]
```

---

## Key Results

### Simulation (N=100, 50 trials, 20% stuck faults)

| Method | MSE | vs. Proposed |
|--------|-----|-------------|
| Average | 4.9 × 10⁻² | 6× worse |
| Trimmed Mean | 2.0 × 10⁻² | 2.4× worse |
| Local Median | 3.1 × 10⁻⁴ (random) / degrades (clustered) | — |
| Plain Consensus | 4.9 × 10⁻² | 6× worse |
| Dist. KF | 4.9 × 10⁻² | 6× worse |
| **Proposed** | **8.2 × 10⁻³** | — |

**Clustered faults (20% density):** Proposed achieves statistically significant 29% MSE improvement over Local Median (0.024 vs. 0.034; 95% CI non-overlapping, 200 trials).

### Practical guidance

| Fault scenario | Recommended method |
|---|---|
| Isolated faults, spatially correlated field | Local Median (lower MSE, 1 comm. round) |
| Clustered or targeted faults | **Proposed** |
| Drift faults | **Proposed** (trend term essential) |

---

## Real-Dataset Highlights

**AirQuality UCI (5 oxide sensors, real drift):**

| Method | MSE |
|--------|-----|
| Average | 0.107 |
| Local Median | 0.255 — collapses under real sensor drift |
| **Proposed** | **0.033** — 7.7× improvement over Average |

**Intel Berkeley Lab (48 temperature sensors):**

| Condition | Local Median | Proposed |
|-----------|-------------|---------|
| Random faults | **0.26 × 10⁻³** (wins) | 0.47 × 10⁻³ |
| Clustered faults | 11.7 × 10⁻³ | **5.2 × 10⁻³** (wins) |

---

## Citation

```bibtex
@article{khemani2026disagreement,
  title   = {Distributed Disagreement-Based Fault Detection and Robust Sensor Fusion via Weighted Consensus},
  author  = {Khemani, Kushal and Kosta, Sujal},
  journal = {IEEE Sensors Letters},
  year    = {2026},
}
```

---

## License

MIT License.
