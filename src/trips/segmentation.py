from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripConfig:
    speed_start_kmh: float = 2.0
    speed_end_kmh: float = 0.0
    end_sustain_minutes: float = 3.0
    min_trip_distance_km: float = 0.1


def energy_consumed_wh(delta_soc_pct: float, usable_ah: float, voltage_v: float) -> float:
    """
    PDF formula: ΔSoC × usable_ah × voltage × 10, where ΔSoC is percentage points.
    """
    if not np.isfinite(delta_soc_pct) or not np.isfinite(usable_ah) or not np.isfinite(voltage_v):
        return float("nan")
    return float(delta_soc_pct) * float(usable_ah) * float(voltage_v) * 10.0


def segment_trips(df: pd.DataFrame, *, cfg: TripConfig = TripConfig()) -> pd.DataFrame:
    """
    Trip start:
      gps_speed_kmh > 2 AND battery_state == 'Discharging' AND device_status == 'Active'
    Trip end:
      gps_speed_kmh == 0 sustained for >3 min OR battery_state == 'Charging'
    Minimum trip distance:
      0.1 km using gps_delta_km sum.
    """
    out = df.copy()
    out["ts"] = pd.to_datetime(out.get("ts"), utc=True, errors="coerce")

    for c in [
        "gps_speed_kmh",
        "gps_delta_km",
        "battery_soc_pct",
        "battery_usable_ah",
        "battery_voltage_v",
        "gps_lat",
        "gps_lon",
    ]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce")

    out["battery_state"] = out.get("battery_state").fillna("").astype(str)
    out["device_status"] = out.get("device_status").fillna("").astype(str)

    out = out.dropna(subset=["device_id", "ts"]).sort_values(["device_id", "ts"], kind="mergesort")

    rows: List[dict] = []

    for device_id, d in out.groupby("device_id", sort=False):
        d = d.reset_index(drop=True)
        if len(d) == 0:
            continue

        speed0 = d["gps_speed_kmh"].fillna(0) <= cfg.speed_end_kmh
        dt = d["ts"].diff().dt.total_seconds().fillna(0).clip(lower=0)
        stop_dur_sec = np.zeros(len(d), dtype=float)
        acc = 0.0
        for i in range(len(d)):
            if bool(speed0.iloc[i]):
                acc += float(dt.iloc[i])
            else:
                acc = 0.0
            stop_dur_sec[i] = acc

        in_trip = False
        start_idx = 0
        trip_num = 0

        for i in range(len(d)):
            speed = d.loc[i, "gps_speed_kmh"]
            batt_state = d.loc[i, "battery_state"]
            status = d.loc[i, "device_status"]

            is_start = (
                np.isfinite(speed)
                and speed > cfg.speed_start_kmh
                and batt_state == "Discharging"
                and status == "Active"
            )
            is_end = (batt_state == "Charging") or (stop_dur_sec[i] >= cfg.end_sustain_minutes * 60.0)

            if not in_trip and is_start:
                in_trip = True
                start_idx = i
                continue

            if in_trip and is_end:
                seg = d.iloc[start_idx : i + 1].copy()
                in_trip = False
                if seg.empty:
                    continue

                dist_km = float(seg["gps_delta_km"].fillna(0).sum())
                if not np.isfinite(dist_km) or dist_km < cfg.min_trip_distance_km:
                    continue

                avg_speed = (
                    float(seg["gps_speed_kmh"].dropna().mean())
                    if seg["gps_speed_kmh"].notna().any()
                    else float("nan")
                )

                soc_start = seg["battery_soc_pct"].iloc[0]
                soc_end = seg["battery_soc_pct"].iloc[-1]
                delta_soc = float(soc_start - soc_end) if np.isfinite(soc_start) and np.isfinite(soc_end) else float("nan")
                delta_soc = max(delta_soc, 0.0) if np.isfinite(delta_soc) else float("nan")

                usable_ah = (
                    float(seg["battery_usable_ah"].dropna().median())
                    if seg["battery_usable_ah"].notna().any()
                    else float("nan")
                )
                voltage_v = (
                    float(seg["battery_voltage_v"].dropna().median())
                    if seg["battery_voltage_v"].notna().any()
                    else float("nan")
                )
                energy_wh = energy_consumed_wh(delta_soc, usable_ah, voltage_v)

                start_lat = float(seg["gps_lat"].dropna().iloc[0]) if seg["gps_lat"].notna().any() else float("nan")
                start_lon = float(seg["gps_lon"].dropna().iloc[0]) if seg["gps_lon"].notna().any() else float("nan")
                end_lat = float(seg["gps_lat"].dropna().iloc[-1]) if seg["gps_lat"].notna().any() else float("nan")
                end_lon = float(seg["gps_lon"].dropna().iloc[-1]) if seg["gps_lon"].notna().any() else float("nan")

                trip_num += 1
                rows.append(
                    {
                        "device_id": device_id,
                        "trip_id": f"{device_id}_{trip_num}",
                        "start_ts": seg["ts"].iloc[0],
                        "end_ts": seg["ts"].iloc[-1],
                        "distance_km": dist_km,
                        "avg_speed_kmh": avg_speed,
                        "energy_consumed_wh": energy_wh,
                        "start_lat": start_lat,
                        "start_lon": start_lon,
                        "end_lat": end_lat,
                        "end_lon": end_lon,
                    }
                )

    return pd.DataFrame(rows)


def label_trips(df: pd.DataFrame, *, cfg: TripConfig = TripConfig()) -> pd.Series:
    """
    Assign trip_id to each telemetry row (NaN when not in a trip) using the same rules as segment_trips().
    """
    out = df.copy()
    out["ts"] = pd.to_datetime(out.get("ts"), utc=True, errors="coerce")

    for c in ["gps_speed_kmh", "battery_soc_pct", "battery_state", "device_status"]:
        if c in out.columns:
            if c in {"battery_state", "device_status"}:
                out[c] = out.get(c).fillna("").astype(str)
            else:
                out[c] = pd.to_numeric(out.get(c), errors="coerce")
        else:
            out[c] = "" if c in {"battery_state", "device_status"} else np.nan

    out = out.dropna(subset=["device_id", "ts"]).sort_values(["device_id", "ts"], kind="mergesort")
    trip_id = pd.Series(pd.NA, index=out.index, dtype="object")

    for device_id, d in out.groupby("device_id", sort=False):
        d = d.reset_index()
        speed0 = d["gps_speed_kmh"].fillna(0) <= cfg.speed_end_kmh
        dt = pd.to_datetime(d["ts"], utc=True, errors="coerce").diff().dt.total_seconds().fillna(0).clip(lower=0)
        stop_dur_sec = np.zeros(len(d), dtype=float)
        acc = 0.0
        for i in range(len(d)):
            if bool(speed0.iloc[i]):
                acc += float(dt.iloc[i])
            else:
                acc = 0.0
            stop_dur_sec[i] = acc

        in_trip = False
        trip_num = 0
        current_id = None
        for i in range(len(d)):
            speed = d.loc[i, "gps_speed_kmh"]
            batt_state = str(d.loc[i, "battery_state"])
            status = str(d.loc[i, "device_status"])

            is_start = (
                np.isfinite(speed)
                and speed > cfg.speed_start_kmh
                and batt_state == "Discharging"
                and status == "Active"
            )
            is_end = (batt_state == "Charging") or (stop_dur_sec[i] >= cfg.end_sustain_minutes * 60.0)

            if not in_trip and is_start:
                in_trip = True
                trip_num += 1
                current_id = f"{device_id}_{trip_num}"

            if in_trip and current_id is not None:
                trip_id.loc[d.loc[i, "index"]] = current_id

            if in_trip and is_end:
                in_trip = False
                current_id = None

    return trip_id.reindex(df.index)
