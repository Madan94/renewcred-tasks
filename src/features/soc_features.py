from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS: List[str] = [
    "soc_delta_5min",
    "discharge_rate_wh",
    "cell_imbalance",
    "temp_deviation",
    "soh_adjusted_cap",
    "charge_headroom",
    "speed_x_soc",
    "rolling_soc_std_1h",
    "idle_energy_drain",
    "rolling_speed_mean_5min",
    "rolling_temp_std_1h",
]


@dataclass(frozen=True)
class SocDataset:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # device_id, ts
    feature_columns: Tuple[str, ...]


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_soc_dataset(
    df: pd.DataFrame,
    *,
    horizon_minutes: int = 10,
    sequence_cadence_seconds: int = 30,
) -> SocDataset:
    """
    Feature engineering for SoC regression.

    Target: battery_soc_pct at t + horizon_minutes.
    Split is done elsewhere; this function only builds the supervised dataset.
    """
    _require_columns(
        df,
        [
            "device_id",
            "ts",
            "battery_soc_pct",
            "battery_usable_ah",
            "battery_voltage_v",
            "capacity_charge_ah",
            "cell_voltage_min",
            "cell_voltage_max",
            "battery_temp_c",
            "battery_soh_pct",
            "gps_speed_kmh",
        ],
    )

    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["device_id", "ts"])
    out = out.sort_values(["device_id", "ts"], kind="mergesort")

    # numeric coercions
    num_cols = [
        "battery_soc_pct",
        "battery_usable_ah",
        "battery_voltage_v",
        "capacity_charge_ah",
        "cell_voltage_min",
        "cell_voltage_max",
        "battery_temp_c",
        "battery_soh_pct",
        "gps_speed_kmh",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # base derived
    out["cell_imbalance"] = out["cell_voltage_max"] - out["cell_voltage_min"]

    step_5min = int((5 * 60) / sequence_cadence_seconds)  # 10 at 30s cadence
    horizon_steps = int((horizon_minutes * 60) / sequence_cadence_seconds)  # 20 at 30s cadence

    g = out.groupby("device_id", sort=False, group_keys=False)

    out["soc_delta_5min"] = g["battery_soc_pct"].diff(step_5min)

    # Energy flow proxy (per PDF): ΔSoC × usable_ah × voltage × 10
    out["delta_soc_5min"] = out["soc_delta_5min"]
    out["discharge_rate_wh"] = (
        out["delta_soc_5min"] * out["battery_usable_ah"] * out["battery_voltage_v"] * 10.0
    )

    # rolling stats using time-aware windows
    def _apply_time_rollings(device_df: pd.DataFrame) -> pd.DataFrame:
        device_df = device_df.set_index("ts", drop=False).sort_index()
        device_df["rolling_7d_mean_temp"] = device_df["battery_temp_c"].rolling("7D", min_periods=10).mean()
        device_df["rolling_soc_std_1h"] = device_df["battery_soc_pct"].rolling("1h", min_periods=10).std()
        device_df["rolling_speed_mean_5min"] = device_df["gps_speed_kmh"].rolling("5min", min_periods=3).mean()
        device_df["rolling_temp_std_1h"] = device_df["battery_temp_c"].rolling("1h", min_periods=10).std()
        return device_df.reset_index(drop=True)

    out = g.apply(_apply_time_rollings)

    out["temp_deviation"] = out["battery_temp_c"] - out["rolling_7d_mean_temp"]
    out["soh_adjusted_cap"] = out["battery_usable_ah"] * (out["battery_soh_pct"] / 100.0)
    out["charge_headroom"] = out["capacity_charge_ah"] / out["battery_usable_ah"]
    out["speed_x_soc"] = out["gps_speed_kmh"] * out["battery_soc_pct"]

    # idle energy drain: discharge proxy only when speed == 0
    out["idle_energy_drain"] = np.where(out["gps_speed_kmh"].fillna(0) <= 0.0, out["discharge_rate_wh"], 0.0)

    # supervised target
    out["target_soc_t_plus"] = out.groupby("device_id", sort=False)["battery_soc_pct"].shift(-horizon_steps)

    meta = out[["device_id", "ts"]].copy()
    X = out[FEATURE_COLUMNS].copy()
    y = out["target_soc_t_plus"].copy()

    # drop rows lacking target
    valid = y.notna()
    meta = meta.loc[valid].reset_index(drop=True)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    # keep NaNs in X for model-specific handling; XGBoost handles NaNs well.
    return SocDataset(X=X, y=y, meta=meta, feature_columns=tuple(FEATURE_COLUMNS))


def time_based_split_per_device(meta: pd.DataFrame, *, test_ratio: float = 0.2) -> pd.Series:
    """
    Returns boolean mask for test rows (True=test) such that per device, the last
    `test_ratio` fraction of timestamps is in the test set.
    """
    _require_columns(meta, ["device_id", "ts"])
    meta2 = meta.copy()
    meta2["ts"] = pd.to_datetime(meta2["ts"], utc=True, errors="coerce")

    def _cutoff(device_df: pd.DataFrame) -> pd.Timestamp:
        device_df = device_df.sort_values("ts", kind="mergesort")
        if len(device_df) == 0:
            return pd.Timestamp.min.tz_localize("UTC")
        idx = int(np.floor((1.0 - test_ratio) * (len(device_df) - 1)))
        idx = max(0, min(idx, len(device_df) - 1))
        return device_df["ts"].iloc[idx]

    cutoffs = meta2.groupby("device_id", sort=False).apply(_cutoff)
    cutoff_map = cutoffs.to_dict()
    return meta2.apply(lambda r: r["ts"] >= cutoff_map.get(r["device_id"], r["ts"]), axis=1).astype(bool)

