"""
datasets.py
-----------
Loaders for four real-world sensor datasets.

Each loader returns a standardised dict:
{
  'Y'          : (N, T) float ndarray  — sensor measurements (normalised)
  'Y_raw'      : (N, T) float ndarray  — sensor measurements (original units)
  'positions'  : (N, 2) float ndarray  — lat/lon or synthetic XY positions
  'sensor_ids' : list[str]
  'timestamps' : list[str]
  'channel'    : str                   — which physical quantity was extracted
  'fault_mask' : (N,) bool or None     — ground-truth faults if available
  'fault_labels': list[str] or None    — fault type strings if available
  'meta'       : dict                  — dataset-specific extras
}

Folder layout expected by all loaders
--------------------------------------
  data/raw/BerkeleyLab.txt
  data/raw/AirQualityUCI.csv
  data/raw/smart_city_sensor_data.csv
  data/raw/sensor_maintenance_data.csv

Call load_<dataset>(path) where path points to the raw file.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(Y: np.ndarray) -> np.ndarray:
    """Z-score normalise each sensor row independently."""
    mu  = np.nanmean(Y, axis=1, keepdims=True)
    std = np.nanstd(Y,  axis=1, keepdims=True)
    std[std < 1e-10] = 1.0
    return (Y - mu) / std


def _fill_nan(Y: np.ndarray) -> np.ndarray:
    """
    Fill NaN values per sensor via linear interpolation, then
    forward/backward fill any remaining edge NaNs.
    """
    Y_out = Y.copy()
    for i in range(Y_out.shape[0]):
        row = Y_out[i]
        nan_mask = np.isnan(row)
        if nan_mask.all():
            Y_out[i] = 0.0
            continue
        if nan_mask.any():
            idx = np.arange(len(row))
            good = ~nan_mask
            f = interp1d(idx[good], row[good], kind='linear',
                         bounds_error=False, fill_value=(row[good][0], row[good][-1]))
            Y_out[i] = f(idx)
    return Y_out


def _synthetic_positions(N: int, seed: int = 0) -> np.ndarray:
    """Random unit-square positions when real coords are unavailable."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (N, 2))


def _latlon_to_unit(coords: np.ndarray) -> np.ndarray:
    """Scale lat/lon array to [0,1]^2."""
    mn, mx = coords.min(axis=0), coords.max(axis=0)
    rng = mx - mn
    rng[rng < 1e-10] = 1.0
    return (coords - mn) / rng


def _pivot_and_clean(df, index_col, columns_col, values_col,
                     max_T=2000, min_obs_frac=0.3):
    """
    Pivot long-format DataFrame to (N_sensors, T) matrix.
    Drops sensors with fewer than min_obs_frac * T valid readings.
    Trims to max_T time steps.

    index_col   → time axis (rows of pivot)
    columns_col → sensor axis (columns of pivot)
    """
    piv = df.pivot_table(index=index_col, columns=columns_col,
                         values=values_col, aggfunc='mean')
    piv = piv.sort_index(axis=0)   # sort time rows

    # Drop sensors (columns) with too few valid readings
    min_obs = int(min_obs_frac * len(piv))
    piv = piv.dropna(thresh=max(1, min_obs), axis=1)

    # Trim to max_T time steps (rows)
    piv = piv.iloc[:max_T]

    Y = piv.values.T.astype(float)     # (N_sensors, T)
    timestamps = [str(x) for x in piv.index.tolist()]
    sensor_ids = [str(x) for x in piv.columns.tolist()]
    return Y, timestamps, sensor_ids


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intel Berkeley Research Lab  (BerkeleyLab.txt)
# ─────────────────────────────────────────────────────────────────────────────

def load_intel_berkeley(path: str, channel: str = "temperature",
                        max_sensors: int = 54, max_T: int = 2000) -> dict:
    """
    Load Intel Berkeley Lab dataset.

    Format (space-separated, no header):
      date  time  epoch  moteid  temperature  humidity  light  voltage

    Parameters
    ----------
    channel : 'temperature' | 'humidity' | 'light' | 'voltage'
    max_T   : maximum time steps to keep (dataset is ~2.3M rows; we
              regularise onto a uniform 30-second grid up to max_T steps)
    """
    path = Path(path)
    print(f"[Berkeley] Loading {path.name} ...")

    cols = ["date", "time", "epoch", "moteid",
            "temperature", "humidity", "light", "voltage"]
    df = pd.read_csv(path, sep=r'\s+', header=None, names=cols,
                     na_values=[''], on_bad_lines='skip',
                     dtype={"moteid": "Int16", "epoch": "Int32",
                            "temperature": float, "humidity": float,
                            "light": float, "voltage": float})

    df = df.dropna(subset=["moteid", channel])
    df["moteid"] = df["moteid"].astype(int)

    # Plausibility filter (remove clearly erroneous readings)
    if channel == "temperature":
        df = df[(df[channel] > -10) & (df[channel] < 60)]
    elif channel == "humidity":
        df = df[(df[channel] > 0)   & (df[channel] < 105)]
    elif channel == "light":
        df = df[(df[channel] >= 0)  & (df[channel] < 5000)]
    elif channel == "voltage":
        df = df[(df[channel] > 1.5) & (df[channel] < 3.5)]

    # Keep sensors with sufficient data
    counts    = df["moteid"].value_counts()
    good_mote = counts[counts >= 2].index.tolist()
    good_mote = sorted(good_mote)[:max_sensors]
    df        = df[df["moteid"].isin(good_mote)]

    # Use epoch as the time index (each epoch ≈ 30 s)
    df_piv = df.groupby(["epoch", "moteid"])[channel].mean().reset_index()
    Y, timestamps, sensor_ids = _pivot_and_clean(
        df_piv, index_col="epoch", columns_col="moteid",
        values_col=channel, max_T=max_T)

    Y     = _fill_nan(Y)
    N, T  = Y.shape
    print(f"[Berkeley] {N} sensors × {T} time steps | channel={channel}")

    # Synthetic positions (real deployment map not in the txt file)
    positions = _synthetic_positions(N, seed=42)

    return dict(
        Y=_normalise(Y), Y_raw=Y,
        positions=positions,
        sensor_ids=sensor_ids,
        timestamps=timestamps,
        channel=channel,
        fault_mask=None, fault_labels=None,
        meta=dict(dataset="Intel Berkeley", source_file=str(path),
                  n_original_sensors=len(good_mote))
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. UCI Air Quality  (AirQualityUCI.csv)
# ─────────────────────────────────────────────────────────────────────────────

def load_air_quality(path: str, channel: str = "multi_sensor",
                     max_T: int = 2000) -> dict:
    """
    Load UCI Air Quality dataset.

    The five metal-oxide sensor columns (PT08.S1 … PT08.S5) are treated as
    a 5-node sensor network measuring a common air-quality index.
    The CO(GT) column is used as a reference signal.
    -200 is the dataset's missing-value sentinel.

    channel : 'oxide_sensors' — uses PT08 sensors as the network
              'gas_sensors'   — uses CO, NMHC, C6H6, NOx, NO2 (ground-truth cols)
    """
    path = Path(path)
    print(f"[AirQuality] Loading {path.name} ...")

    # Auto-detect separator (UCI dataset distributed with both ; and , variants)
    with open(path, 'r', encoding='utf-8-sig') as _f:
        _first = _f.readline()
    _sep = ';' if ';' in _first else ','
    _decimal = ',' if _sep == ';' else '.'
    df = pd.read_csv(path, sep=_sep, decimal=_decimal, na_values=[-200, '-200'])
    if 'Date' in df.columns and 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            dayfirst=True, errors='coerce')
        df = df.drop(columns=['Date', 'Time'], errors='ignore')
    elif 'datetime' not in df.columns:
        df['datetime'] = pd.RangeIndex(len(df))
    df = df.dropna(how='all')

    oxide_cols = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
                  "PT08.S4(NO2)", "PT08.S5(O3)"]
    gas_cols   = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

    if channel == "oxide_sensors":
        sensor_cols = [c for c in oxide_cols if c in df.columns]
    else:
        sensor_cols = [c for c in gas_cols if c in df.columns]

    df_sub = df[["datetime"] + sensor_cols].copy()
    df_sub = df_sub.replace(-200, np.nan).iloc[:max_T]
    df_sub = df_sub.set_index("datetime")

    Y          = df_sub.values.T.astype(float)   # (N_sensors, T)
    Y          = _fill_nan(Y)
    N, T       = Y.shape
    sensor_ids = sensor_cols
    timestamps = [str(x) for x in df_sub.index.tolist()]

    print(f"[AirQuality] {N} sensors × {T} time steps | channel={channel}")

    # Drift detection: flag sensors whose mean changes significantly
    # in the second half vs first half of the time series
    half = T // 2
    drift_ratio = (np.abs(Y[:, half:].mean(axis=1) - Y[:, :half].mean(axis=1))
                   / (Y[:, :half].std(axis=1) + 1e-8))
    fault_mask   = drift_ratio > 1.5
    fault_labels = ["drift" if fault_mask[i] else "none"
                    for i in range(N)]

    positions = _synthetic_positions(N, seed=1)

    return dict(
        Y=_normalise(Y), Y_raw=Y,
        positions=positions,
        sensor_ids=sensor_ids,
        timestamps=timestamps,
        channel=channel,
        fault_mask=fault_mask,
        fault_labels=fault_labels,
        meta=dict(dataset="UCI Air Quality", source_file=str(path),
                  drift_ratios=drift_ratio.tolist())
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Smart City IoT  (smart_city_sensor_data.csv)
# ─────────────────────────────────────────────────────────────────────────────

# Column mapping for Turkish headers
_SMART_CITY_COLS = {
    "Şehir ID'si/Adı":      "city",
    "Sensör ID'si/Adı":     "sensor_id",
    "Enlem":                "lat",
    "Boylam":               "lon",
    "Tarih/Zaman":          "datetime",
    "Sensör Tipi":          "sensor_type",
    "Araç Sayısı":          "vehicle_count",
    "kWh":                  "energy_kwh",
    "Doluluk Oranı":        "occupancy",
    "Gürültü Seviyesi":     "noise_db",
}

# Fallback: positional column names if headers are garbled
_SMART_CITY_POS = [
    "city", "sensor_id", "lat", "lon", "datetime",
    "sensor_type", "street_type", "nearby_services",
    "vehicle_count", "energy_kwh", "occupancy", "noise_db"
]


def load_smart_city(path: str, sensor_type_filter: str = "all",
                    channel: str = "auto", max_T: int = 2000,
                    city_filter: str = "all") -> dict:
    """
    Load Smart City IoT dataset.

    The dataset contains multiple sensor types in multiple cities.
    We extract one channel per sensor type and build a (N, T) matrix
    where each row is one sensor and each column is one hour.

    Parameters
    ----------
    sensor_type_filter : 'traffic' | 'energy' | 'waste' | 'environment' | 'all'
    channel            : 'auto' selects the natural channel for each sensor type
    city_filter        : city name substring (e.g. 'Istanbul') or 'all'
    """
    path = Path(path)
    print(f"[SmartCity] Loading {path.name} ...")

    # Try with header first; if garbled, fall back to positional
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip')
        # Rename Turkish cols if present
        df = df.rename(columns={k: v for k, v in _SMART_CITY_COLS.items()
                                  if k in df.columns})
        # If still no recognised cols, try positional
        if "sensor_id" not in df.columns and len(df.columns) >= 8:
            df.columns = _SMART_CITY_POS[:len(df.columns)]
    except Exception:
        df = pd.read_csv(path, header=None, names=_SMART_CITY_POS,
                         encoding='utf-8-sig', on_bad_lines='skip',
                         skiprows=1)

    df = df.dropna(how='all')

    # City filter
    if city_filter != "all" and "city" in df.columns:
        df = df[df["city"].astype(str).str.contains(city_filter, case=False, na=False)]

    # Sensor type filter & channel selection
    type_channel_map = {
        "traffic":     ("vehicle_count",  ["vehicle_count"]),
        "energy":      ("energy_kwh",     ["energy_kwh"]),
        "waste":       ("occupancy",      ["occupancy"]),
        "environment": ("noise_db",       ["noise_db"]),
    }

    if sensor_type_filter != "all" and "sensor_type" in df.columns:
        kw = {"traffic": "Trafik", "energy": "Enerji",
              "waste": "Atık", "environment": "Çevre"}.get(sensor_type_filter, sensor_type_filter)
        df = df[df["sensor_type"].astype(str).str.contains(kw, case=False, na=False)]
        ch, candidate_cols = type_channel_map.get(sensor_type_filter, ("auto", []))
    else:
        # Auto: pick the first numeric column with most non-null values
        candidate_cols = ["vehicle_count", "energy_kwh", "occupancy", "noise_db"]
        ch = "mixed"

    # Find the best numeric column
    value_col = None
    for col in candidate_cols:
        if col in df.columns:
            n_valid = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if n_valid > 10:
                value_col = col
                break

    if value_col is None:
        # Last resort: first numeric column
        for col in df.select_dtypes(include=[np.number]).columns:
            value_col = col
            break

    if value_col is None:
        raise ValueError(f"No usable numeric column found in {path.name}")

    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    # Parse datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"],
                                         dayfirst=True, errors='coerce')
        df = df.dropna(subset=["datetime"])
        df["t_idx"] = df["datetime"].rank(method='dense').fillna(0).astype(int)
    else:
        df["t_idx"] = np.arange(len(df))

    sid_col = "sensor_id" if "sensor_id" in df.columns else df.columns[1]
    df_piv = df.groupby(["t_idx", sid_col])[value_col].mean().reset_index()

    Y, timestamps, sensor_ids = _pivot_and_clean(
        df_piv, index_col="t_idx", columns_col=sid_col,
        values_col=value_col, max_T=max_T, min_obs_frac=0.2)

    Y    = _fill_nan(Y)
    N, T = Y.shape
    print(f"[SmartCity] {N} sensors × {T} time steps | channel={value_col}")

    # Extract positions if available
    if "lat" in df.columns and "lon" in df.columns:
        pos_df = df.groupby(sid_col)[["lat", "lon"]].first().reindex(sensor_ids)
        coords = pos_df[["lat", "lon"]].values.astype(float)
        nan_rows = np.isnan(coords).any(axis=1)
        if not nan_rows.all():
            coords[nan_rows] = np.nanmean(coords[~nan_rows], axis=0)
            positions = _latlon_to_unit(coords)
        else:
            positions = _synthetic_positions(N)
    else:
        positions = _synthetic_positions(N)

    return dict(
        Y=_normalise(Y), Y_raw=Y,
        positions=positions,
        sensor_ids=sensor_ids,
        timestamps=timestamps,
        channel=value_col,
        fault_mask=None, fault_labels=None,
        meta=dict(dataset="Smart City IoT", source_file=str(path),
                  sensor_type=sensor_type_filter, city=city_filter)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sensor Maintenance  (sensor_maintenance_data.csv)
# ─────────────────────────────────────────────────────────────────────────────

_MAINT_COL_MAP = {
    "Sensor_ID":          "sensor_id",
    "Timestamp":          "datetime",
    "Voltage (V)":        "voltage",
    "Current (A)":        "current",
    "Temperature (°C)":   "temperature",
    "Temperature (Â°C)":  "temperature",    # encoding variant
    "Power (W)":          "power",
    "Humidity (%)":       "humidity",
    "Vibration (m/s2)":   "vibration",
    "Vibration (m/sÂ2)":  "vibration",
    "Equipment_ID":       "equipment_id",
    "Operational Status": "op_status",
    "Fault Status":       "fault_status",
    "Failure Type":       "failure_type",
}

def load_sensor_maintenance(path: str, channel: str = "temperature",
                             max_T: int = 500) -> dict:
    """
    Load Sensor Maintenance dataset.

    Each row is one unique sensor at one unique timestamp (cross-sectional).
    Since no sensor is observed over multiple time steps, we treat the
    numeric measurement channels (voltage, current, temperature, power,
    humidity, vibration) as N=6 parallel 'sensors' over T time steps,
    sorted by timestamp. Ground-truth fault labels come from 'Fault Status'.

    Parameters
    ----------
    channel : ignored (all channels used as sensor dimensions)
    max_T   : maximum number of time steps to keep
    """
    path = Path(path)
    print(f"[Maintenance] Loading {path.name} ...")

    df = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip')
    df = df.rename(columns={k: v for k, v in _MAINT_COL_MAP.items()
                              if k in df.columns})
    df = df.dropna(how='all')

    # Normalise fault status
    if "fault_status" in df.columns:
        df["fault_status"] = (df["fault_status"]
                               .astype(str).str.strip().str.lower()
                               .map(lambda x: "fault"
                                    if "fault" in x and "no fault" not in x
                                    else "none"))
    else:
        df["fault_status"] = "none"

    if "failure_type" in df.columns:
        df["failure_type"] = (df["failure_type"]
                               .astype(str).str.strip()
                               .replace({"None": "none", "nan": "none"}))
    else:
        df["failure_type"] = "none"

    # Sort by datetime or original row order
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"],
                                         dayfirst=True, errors='coerce')
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df = df.iloc[:max_T]

    # Use numeric measurement channels as sensor dimensions (N channels × T rows)
    meas_cols = [c for c in ["voltage", "current", "temperature",
                              "power", "humidity", "vibration"]
                 if c in df.columns]
    if not meas_cols:
        meas_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]

    Y = np.column_stack([pd.to_numeric(df[c], errors='coerce').values
                         for c in meas_cols]).T.astype(float)  # (N_channels, T)
    Y = _fill_nan(Y)
    N, T = Y.shape

    sensor_ids = meas_cols
    timestamps = ([str(x) for x in df["datetime"].tolist()]
                  if "datetime" in df.columns
                  else [str(i) for i in range(T)])

    # Ground-truth fault labels per time step (one per row in this dataset)
    fault_per_step = (df["fault_status"] == "fault").values  # (T,) bool

    # A measurement channel is "faulty" if its values look anomalous when
    # the row is labelled fault vs healthy (z-score separation > 1)
    fault_mask = np.zeros(N, dtype=bool)
    for i, col in enumerate(meas_cols):
        faulty_vals  = Y[i, fault_per_step]
        healthy_vals = Y[i, ~fault_per_step]
        if len(faulty_vals) > 0 and len(healthy_vals) > 0:
            sep = (abs(faulty_vals.mean() - healthy_vals.mean())
                   / (healthy_vals.std() + 1e-8))
            fault_mask[i] = sep > 1.0

    fault_labels = ["fault" if f else "none" for f in fault_mask]
    positions    = _synthetic_positions(N, seed=7)

    print(f"[Maintenance] {N} channels × {T} time steps | "
          f"fault rows: {fault_per_step.sum()}/{T} "
          f"({fault_per_step.mean():.0%})")

    failure_type_counts = df["failure_type"].value_counts().to_dict()

    return dict(
        Y=_normalise(Y), Y_raw=Y,
        positions=positions,
        sensor_ids=sensor_ids,
        timestamps=timestamps,
        channel="multi-channel",
        fault_mask=fault_mask,
        fault_labels=fault_labels,
        meta=dict(dataset="Sensor Maintenance", source_file=str(path),
                  fault_fraction=float(fault_per_step.mean()),
                  failure_types=failure_type_counts,
                  row_fault_labels=fault_per_step.tolist())
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unified loader dispatcher
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "berkeley":    (load_intel_berkeley,    "BerkeleyLab.txt"),
    "airquality":  (load_air_quality,       "AirQualityUCI.csv"),
    "smartcity":   (load_smart_city,        "smart_city_sensor_data.csv"),
    "maintenance": (load_sensor_maintenance,"sensor_maintenance_data.csv"),
}

def load_dataset(name: str, data_dir: str = "data/raw", **kwargs) -> dict:
    """
    Convenience dispatcher.

    Usage
    -----
        data = load_dataset("berkeley", channel="temperature")
        data = load_dataset("airquality", channel="oxide_sensors")
        data = load_dataset("smartcity", sensor_type_filter="traffic")
        data = load_dataset("maintenance", channel="temperature")
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")
    loader, filename = DATASET_REGISTRY[name]
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Place '{filename}' in the '{data_dir}/' folder.")
    return loader(str(path), **kwargs)
