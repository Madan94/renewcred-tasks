from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanParams:
    process_var: float = 1e-6
    measurement_var: float = 1e-4


def _kalman_1d(z: np.ndarray, q: float, r: float) -> np.ndarray:
    """
    Simple 1D Kalman filter for random-walk model:
      x_t = x_{t-1} + w,  w~N(0,q)
      z_t = x_t + v,      v~N(0,r)
    """
    x = np.empty_like(z, dtype=float)
    p = 1.0
    x_hat = float("nan")
    for j in range(len(z)):
        if np.isfinite(z[j]):
            x_hat = float(z[j])
            break
    if not np.isfinite(x_hat):
        x_hat = 0.0
    for i in range(len(z)):
        # predict
        p = p + q
        # update
        if np.isfinite(z[i]):
            k = p / (p + r)
            x_hat = x_hat + k * (z[i] - x_hat)
            p = (1 - k) * p
        x[i] = x_hat
    return x


def smooth_latlon_kalman(
    df: pd.DataFrame,
    *,
    params: KalmanParams = KalmanParams(),
) -> pd.DataFrame:
    """
    Stage 2: per-device smoothing on (lat, lon).
    Adds columns: gps_lat_smooth, gps_lon_smooth
    """
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out["gps_lat"] = pd.to_numeric(out.get("gps_lat"), errors="coerce")
    out["gps_lon"] = pd.to_numeric(out.get("gps_lon"), errors="coerce")
    out = out.dropna(subset=["device_id", "ts"]).sort_values(["device_id", "ts"], kind="mergesort")

    def _smooth_device(d: pd.DataFrame) -> pd.DataFrame:
        lat = d["gps_lat"].to_numpy(dtype=float)
        lon = d["gps_lon"].to_numpy(dtype=float)
        # keep dropouts as NaN; filter handles them by skipping update
        lat_s = _kalman_1d(lat, params.process_var, params.measurement_var)
        lon_s = _kalman_1d(lon, params.process_var, params.measurement_var)
        d = d.copy()
        d["gps_lat_smooth"] = lat_s
        d["gps_lon_smooth"] = lon_s
        return d

    out = out.groupby("device_id", sort=False, group_keys=False).apply(_smooth_device)
    return out

