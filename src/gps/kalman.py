from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanParams:
    process_var: float = 1e-6
    measurement_var: float = 1e-4


def _kalman_2d(lat: np.ndarray, lon: np.ndarray, q: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple 2D Kalman filter for a random-walk position model:
      x_t = x_{t-1} + w,  w~N(0,Q)
      z_t = x_t + v,      v~N(0,R)

    State: x = [lat, lon]
    Q and R are diagonal with variances (q, q) and (r, r).
    Handles NaNs by performing a partial update on available measurement dimensions.
    """
    n = int(min(len(lat), len(lon)))
    lat_s = np.empty(n, dtype=float)
    lon_s = np.empty(n, dtype=float)

    # initial state from first finite measurement
    x = np.array([0.0, 0.0], dtype=float)
    init = False
    for i in range(n):
        if np.isfinite(lat[i]) and np.isfinite(lon[i]):
            x[:] = [float(lat[i]), float(lon[i])]
            init = True
            break
    if not init:
        for i in range(n):
            if np.isfinite(lat[i]):
                x[0] = float(lat[i])
                init = True
                break
        for i in range(n):
            if np.isfinite(lon[i]):
                x[1] = float(lon[i])
                init = True
                break

    P = np.eye(2, dtype=float)  # initial covariance
    Q = np.eye(2, dtype=float) * float(q)
    R_full = np.eye(2, dtype=float) * float(r)

    for i in range(n):
        # predict
        P = P + Q

        z_lat = lat[i]
        z_lon = lon[i]
        have_lat = np.isfinite(z_lat)
        have_lon = np.isfinite(z_lon)

        if have_lat or have_lon:
            if have_lat and have_lon:
                z = np.array([float(z_lat), float(z_lon)], dtype=float)
                H = np.eye(2, dtype=float)
                R = R_full
            elif have_lat:
                z = np.array([float(z_lat)], dtype=float)
                H = np.array([[1.0, 0.0]], dtype=float)
                R = np.array([[float(r)]], dtype=float)
            else:
                z = np.array([float(z_lon)], dtype=float)
                H = np.array([[0.0, 1.0]], dtype=float)
                R = np.array([[float(r)]], dtype=float)

            # innovation
            y = z - (H @ x)
            S = H @ P @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            x = x + (K @ y)
            P = (np.eye(2, dtype=float) - (K @ H)) @ P

        lat_s[i] = float(x[0])
        lon_s[i] = float(x[1])

    return lat_s, lon_s


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
        lat_s, lon_s = _kalman_2d(lat, lon, params.process_var, params.measurement_var)
        d = d.copy()
        d["gps_lat_smooth"] = lat_s
        d["gps_lon_smooth"] = lon_s
        return d

    out = out.groupby("device_id", sort=False, group_keys=False).apply(_smooth_device)
    return out

