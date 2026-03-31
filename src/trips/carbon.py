from __future__ import annotations

import numpy as np
import pandas as pd


def net_tco2e_kg(energy_wh: float, distance_km: float) -> float:
    """net tCO2e = (energy_wh / 1000) * 0.716 - distance_km * 0.000150"""
    if not np.isfinite(energy_wh) or not np.isfinite(distance_km):
        return float("nan")
    return float(energy_wh) / 1000.0 * 0.716 - float(distance_km) * 0.000150


def trips_with_carbon_credits(trips: pd.DataFrame) -> pd.DataFrame:
    out = trips.copy()
    e = pd.to_numeric(out.get("energy_consumed_wh"), errors="coerce")
    d = pd.to_numeric(out.get("distance_km"), errors="coerce")
    out["net_tco2e"] = (e / 1000.0) * 0.716 - d * 0.000150
    return out
