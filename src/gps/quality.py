from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_KM * c


@dataclass(frozen=True)
class GPSAnomalyMasks:
    position_jump: pd.Series
    signal_dropout: pd.Series
    coordinate_freeze: pd.Series


def detect_gps_anomalies(df: pd.DataFrame) -> Tuple[pd.DataFrame, GPSAnomalyMasks]:
    """
    Stage 1 anomalies (per PDF):
    (a) Position jump: Δdistance > 1km in <30 sec
    (b) Signal dropout: gps_lat=0.0 and gps_lon=0.0
    (c) Coordinate freeze: same lat/lon for >5 consecutive pings while gps_speed>2
    """
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    for c in ["gps_lat", "gps_lon", "gps_speed_kmh"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan

    out = out.dropna(subset=["device_id", "ts"]).sort_values(["device_id", "ts"], kind="mergesort")

    g = out.groupby("device_id", sort=False, group_keys=False)

    out["prev_lat"] = g["gps_lat"].shift(1)
    out["prev_lon"] = g["gps_lon"].shift(1)
    out["prev_ts"] = g["ts"].shift(1)

    dt = (out["ts"] - out["prev_ts"]).dt.total_seconds()
    dt = dt.where(dt.notna(), np.nan)

    valid_pair = out["gps_lat"].notna() & out["gps_lon"].notna() & out["prev_lat"].notna() & out["prev_lon"].notna()
    dist = pd.Series(np.nan, index=out.index, dtype="float64")
    if valid_pair.any():
        dist.loc[valid_pair] = haversine_km(
            out.loc[valid_pair, "prev_lat"].values,
            out.loc[valid_pair, "prev_lon"].values,
            out.loc[valid_pair, "gps_lat"].values,
            out.loc[valid_pair, "gps_lon"].values,
        )

    out["delta_km_haversine"] = dist
    out["delta_sec"] = dt

    position_jump = (out["delta_km_haversine"] > 1.0) & (out["delta_sec"] < 30.0)

    signal_dropout = (out["gps_lat"] == 0.0) & (out["gps_lon"] == 0.0)

    lat = out["gps_lat"].to_numpy()
    lon = out["gps_lon"].to_numpy()
    spd = out["gps_speed_kmh"].to_numpy()
    did = out["device_id"].to_numpy()
    n = len(out)
    freeze_arr = np.zeros(n, dtype=bool)
    run = 0
    prev_dev = object()
    for i in range(n):
        if did[i] != prev_dev:
            prev_dev = did[i]
            run = 0
        if i == 0 or did[i] != did[i - 1]:
            continue
        same = (
            np.isfinite(lat[i])
            and np.isfinite(lon[i])
            and np.isfinite(lat[i - 1])
            and np.isfinite(lon[i - 1])
            and lat[i] == lat[i - 1]
            and lon[i] == lon[i - 1]
        )
        moving = np.isfinite(spd[i]) and spd[i] > 2.0
        if same and moving:
            run += 1
        else:
            run = 0
        if run >= 5:
            freeze_arr[i] = True
    coordinate_freeze = pd.Series(freeze_arr, index=out.index, dtype=bool)

    masks = GPSAnomalyMasks(
        position_jump=position_jump.astype(bool),
        signal_dropout=signal_dropout.astype(bool),
        coordinate_freeze=coordinate_freeze.astype(bool),
    )

    out["gps_anomaly_position_jump"] = masks.position_jump
    out["gps_anomaly_signal_dropout"] = masks.signal_dropout
    out["gps_anomaly_coordinate_freeze"] = masks.coordinate_freeze
    return out, masks

