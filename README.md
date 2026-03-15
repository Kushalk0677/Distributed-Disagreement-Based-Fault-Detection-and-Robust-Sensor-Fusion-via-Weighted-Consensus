# Neighbor Disagreement-Based Fault Detection for Distributed Sensor Fusion

Simulation code accompanying the IEEE Sensors Letters paper:

> **"Distributed Disagreement-Based Fault Detection andRobust Sensor Fusion via Weighted Consensus"**

---

## Overview

This repository implements a lightweight distributed fault detection mechanism for sensor networks. Each sensor computes a *disagreement score* against its neighbors' median measurement and uses it as a reliability weight during fusion. No centralized controller, prior noise statistics, or explicit fault models are required.

---

## Repository Structure

```
sensor_fusion_repo/
│
├── src/
│   ├── network.py       # Random geometric graph topology builder
│   ├── signal.py        # Ground-truth signal + fault injection models
│   ├── fusion.py        # Fusion algorithms (average, median, proposed)
│   ├── metrics.py       # MSE, RMSE, FDR, FAR evaluation
│   └── visualize.py     # All paper figures
│
├── experiments/
│   └── run_experiments.py   # Main experiment runner (all 5 experiments)
│
├── configs/
│   └── default.yaml         # All hyperparameters in one place
│
├── results/
│   ├── figures/             # PDF figures (generated on run)
│   └── summary.json         # Numeric results (generated on run)
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/sensor_fusion_repo.git
cd sensor_fusion_repo

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running Experiments

### Full experiments (all 5, ~5 min)
```bash
python experiments/run_experiments.py
```

### Quick mode (reduced trials, ~30 sec)
```bash
python experiments/run_experiments.py --quick
```

### Custom config
```bash
python experiments/run_experiments.py --config configs/default.yaml
```

---


## Configuration

All parameters are in `configs/default.yaml`:

```yaml
network:
  N: 30          # number of sensors
  radius: 0.35   # communication radius

signal:
  T: 300         # time steps
  type: sinusoid # sinusoid | step | ramp | composite
  noise_std: 0.1

faults:
  fault_fraction: 0.20
  fault_type: stuck   # stuck | drift | malicious
  fault_onset: 50

fusion:
  alpha: 0.9     # temporal smoothing coefficient
```

---

## Fault Models

| Type | Description |
|------|-------------|
| `stuck` | Sensor freezes at its value at fault onset |
| `drift` | Slowly increasing bias after fault onset |
| `malicious` | Replaced with uniform random values |

---

## Method Summary

The proposed method (equations from the paper):

1. **Disagreement metric:** `d_i(t) = |y_i(t) − median(y_j(t), j ∈ N_i)|`
2. **Temporal smoothing:** `D_i(t) = α·D_i(t−1) + (1−α)·d_i(t)`
3. **Reliability weight:** `w_i(t) = 1 / (1 + D_i(t))`
4. **Weighted fusion:** `ŝ(t) = Σ w_i·y_i / Σ w_i`

---

## Citation

```bibtex
@article{yourname2024disagreement,
  title   = {Neighbor Disagreement-Based Fault Detection for Distributed Sensor Fusion},
  author  = {Author Name},
  journal = {IEEE Sensors Letters},
  year    = {2024},
}
```

---

## Real-World Datasets

Place data files in `data/raw/` (see `data/raw/README_DATA.md` for download links and format notes).

```
data/raw/
  BerkeleyLab.txt               ← Intel Berkeley Lab (54 sensors, temp/humidity/light/voltage)
  AirQualityUCI.csv             ← UCI Air Quality (5 oxide sensors, known drift)
  smart_city_sensor_data.csv    ← Smart City IoT (traffic/energy/noise, lat/lon)
  sensor_maintenance_data.csv   ← Sensor Maintenance (ground-truth fault labels)
```

### Running on real data

```bash
# List datasets and check which files are present
python experiments/run_real_data.py --list

# Run all available datasets
python experiments/run_real_data.py

# Run one specific dataset
python experiments/run_real_data.py --dataset berkeley
python experiments/run_real_data.py --dataset airquality
python experiments/run_real_data.py --dataset smartcity
python experiments/run_real_data.py --dataset maintenance
```

Output figures are saved to `results/figures/real_<dataset>_*.pdf`.
A JSON summary is saved to `results/real_data_summary.json`.

### What each dataset tests

| Dataset             | Key experiment                                  | Ground truth |
|---------------------|-------------------------------------------------|--------------|
| Berkeley            | Temperature fusion across 54 sensors            | None         |
| Air Quality         | Oxide sensor drift detection                    | Heuristic    |
| Smart City IoT      | Traffic / energy / noise by city and type       | None         |
| Sensor Maintenance  | Fault detection with labelled fault status      | ✓ Yes        |
