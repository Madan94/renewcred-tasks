from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.sort_values(["device_id", "ts"], kind="mergesort").reset_index(drop=True)

    for c in [
        "battery_soc_pct",
        "battery_usable_ah",
        "battery_voltage_v",
        "battery_soh_pct",
        "battery_temp_c",
        "cell_voltage_min",
        "cell_voltage_max",
        "gps_speed_kmh",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["cell_imbalance"] = out["cell_voltage_max"] - out["cell_voltage_min"]

    def _time_align_delta(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values("ts", kind="mergesort").copy()
        right = d[["ts", "battery_soc_pct"]].rename(columns={"ts": "ts_ref", "battery_soc_pct": "soc_ref"})
        left = d[["ts"]].copy()
        left["ts_ref"] = left["ts"] - pd.Timedelta(minutes=5)
        lag = pd.merge_asof(
            left.sort_values("ts_ref"),
            right.sort_values("ts_ref"),
            on="ts_ref",
            direction="backward",
        )["soc_ref"].to_numpy()
        d["soc_delta_5min"] = d["battery_soc_pct"].to_numpy() - lag
        return d

    out = out.groupby("device_id", sort=False, group_keys=False).apply(_time_align_delta)
    out["discharge_rate_wh"] = out["soc_delta_5min"] * out["battery_usable_ah"] * out["battery_voltage_v"] * 10.0
    out["discharge_rate_wh_abs"] = out["discharge_rate_wh"].abs()

    def _roll(x: pd.DataFrame) -> pd.DataFrame:
        x = x.set_index("ts", drop=False).sort_index()
        # Use absolute magnitude for anomaly checks (spikes can occur in either direction).
        x["rolling_discharge_mean_1h"] = x["discharge_rate_wh_abs"].rolling("1h", min_periods=5).mean()
        x["soh_rolling_max_7d"] = x["battery_soh_pct"].rolling("7D", min_periods=10).max()
        return x.reset_index(drop=True)

    out = out.groupby("device_id", sort=False, group_keys=False).apply(_roll)
    return out


def isolation_forest_confidence(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(Xs)
    s = iso.decision_function(Xs)
    s_min, s_max = float(s.min()), float(s.max())
    if s_max - s_min < 1e-9:
        return np.zeros_like(s)
    conf = 1.0 - (s - s_min) / (s_max - s_min)
    return np.clip(conf, 0.0, 1.0)


def build_anomaly_flags_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output CSV: device_id, ts, anomaly_type, confidence_score, carbon_credit_impact_pct
    """
    out = _prepare(df)
    n = len(out)
    if n == 0:
        return pd.DataFrame(
            columns=["device_id", "ts", "anomaly_type", "confidence_score", "carbon_credit_impact_pct"]
        )

    feat = np.column_stack(
        [
            out["cell_imbalance"].fillna(0).values,
            out["battery_soh_pct"].fillna(out["battery_soh_pct"].median()).values,
            out["battery_temp_c"].fillna(out["battery_temp_c"].median()).values,
            out["battery_soc_pct"].fillna(out["battery_soc_pct"].median()).values,
        ]
    )
    if_conf = isolation_forest_confidence(feat)

    # Used as a weak proxy for SoC volatility; only used for impact scaling.
    soc_delta = out.groupby("device_id", sort=False)["battery_soc_pct"].diff().abs().fillna(0)

    rows = []
    for i in range(n):
        if_conf_i = float(if_conf[i])
        # Impact is only non-zero for anomalies that can inflate energy / SoC-delta based crediting.
        # Keep as a conservative estimate in [0,100].
        cc_base = min(100.0, float(soc_delta.iloc[i]) * 0.3 + if_conf_i * 25.0)

        if out["cell_imbalance"].iloc[i] > 0.05:
            rows.append(
                {
                    "device_id": out["device_id"].iloc[i],
                    "ts": out["ts"].iloc[i],
                    "anomaly_type": "cell_imbalance_high",
                    "confidence_score": round(if_conf_i, 4),
                    # Voltage imbalance is a health risk but doesn't necessarily inflate SoC-delta.
                    "carbon_credit_impact_pct": 0.0,
                }
            )
        if (
            out["soh_rolling_max_7d"].notna().iloc[i]
            and out["battery_soh_pct"].notna().iloc[i]
            and (out["soh_rolling_max_7d"].iloc[i] - out["battery_soh_pct"].iloc[i]) > 1.0
        ):
            rows.append(
                {
                    "device_id": out["device_id"].iloc[i],
                    "ts": out["ts"].iloc[i],
                    "anomaly_type": "soh_drop_7d",
                    "confidence_score": round(if_conf_i, 4),
                    # SoH impacts capacity assumptions over longer horizons; set low direct credit inflation.
                    "carbon_credit_impact_pct": round(min(25.0, cc_base), 2),
                }
            )
        rdm = out["rolling_discharge_mean_1h"].iloc[i]
        dr = out["discharge_rate_wh_abs"].iloc[i]
        if pd.notna(rdm) and pd.notna(dr) and rdm > 0 and dr > 3.0 * rdm:
            # Estimate potential over-credit if a spike inflates discharge proxy vs baseline.
            over = (float(dr) / float(rdm)) - 1.0
            spike_impact = float(np.clip(over * 100.0, 0.0, 100.0))
            rows.append(
                {
                    "device_id": out["device_id"].iloc[i],
                    "ts": out["ts"].iloc[i],
                    "anomaly_type": "discharge_spike_vs_rolling",
                    "confidence_score": round(if_conf_i, 4),
                    "carbon_credit_impact_pct": round(max(cc_base, spike_impact), 2),
                }
            )
        if pd.notna(out["battery_temp_c"].iloc[i]) and out["battery_temp_c"].iloc[i] > 40.0:
            rows.append(
                {
                    "device_id": out["device_id"].iloc[i],
                    "ts": out["ts"].iloc[i],
                    "anomaly_type": "temperature_spike",
                    "confidence_score": round(if_conf_i, 4),
                    # Temperature spikes can correlate with sensor drift; keep a moderate conservative impact.
                    "carbon_credit_impact_pct": round(min(50.0, cc_base), 2),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["device_id", "ts", "anomaly_type"]).reset_index(drop=True)
