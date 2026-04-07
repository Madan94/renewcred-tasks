# src/ingestion/pipeline.py

import hashlib
import json
import logging
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.config import SECRET_SALT

logging.basicConfig(level=logging.INFO)


EXPECTED_COLUMNS: List[str] = [
    "device_id",
    "imei_token",
    "last_ping_time",
    "device_status",
    "gps_lat",
    "gps_lon",
    "gps_speed_kmh",
    "gps_delta_km",
    "gps_total_km",
    "battery_state",
    "battery_soc_pct",
    "battery_capacity_ah",
    "battery_usable_ah",
    "capacity_discharge_ah",
    "capacity_charge_ah",
    "battery_voltage_v",
    "cell_voltage_min",
    "cell_voltage_max",
    "battery_temp_c",
    "battery_soh_pct",
    "ts",
]

_STRING_COLUMNS = {"device_id", "imei_token", "device_status", "battery_state"}
_NUMERIC_COLUMNS = [
    "gps_lat",
    "gps_lon",
    "gps_speed_kmh",
    "gps_delta_km",
    "gps_total_km",
    "battery_soc_pct",
    "battery_capacity_ah",
    "battery_usable_ah",
    "capacity_discharge_ah",
    "capacity_charge_ah",
    "battery_voltage_v",
    "cell_voltage_min",
    "cell_voltage_max",
    "battery_temp_c",
    "battery_soh_pct",
]


def _hash_identifier(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bytes):
        value_str = value.decode("utf-8", errors="ignore")
    else:
        value_str = str(value)
    return hashlib.sha256((value_str + SECRET_SALT).encode("utf-8")).hexdigest()


def safe_json_load(payload: Any) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    if isinstance(payload, float) and pd.isna(payload):
        return None
    if not isinstance(payload, str):
        payload = str(payload)
    try:
        parsed = json.loads(payload)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def extract_fields(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract canonical 21-column record from the nested MQTT payload.
    Malformed JSON and missing nested objects must never crash parsing.
    """
    if not isinstance(data, dict):
        return None

    payload = data.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    raw_device_id = payload.get("deviceId")
    raw_imei = payload.get("imei")

    # If both primary identifiers are missing, we cannot reliably anonymize; drop row.
    if raw_device_id is None and raw_imei is None:
        return None

    gps = payload.get("gps", {})
    if not isinstance(gps, dict):
        gps = {}

    battery = payload.get("battery", {})
    if not isinstance(battery, dict):
        battery = {}

    # Requirement: never store raw identifiers. Prefer hashing deviceId; if absent, fall back to IMEI hash.
    anon_device_id = _hash_identifier(raw_device_id) or _hash_identifier(raw_imei)
    return {
        # Requirement: anonymize device IDs and IMEI values (never store raw PII).
        "device_id": anon_device_id,
        "imei_token": _hash_identifier(raw_imei),
        "last_ping_time": payload.get("lastPingTime"),
        "device_status": payload.get("status"),
        "gps_lat": gps.get("gpsLatitude"),
        "gps_lon": gps.get("gpsLongitude"),
        "gps_speed_kmh": gps.get("gpsGroundSpeed"),
        "gps_delta_km": gps.get("gpsGroundDeltaDistance"),
        "gps_total_km": gps.get("gpsTotalGroundDistance"),
        "battery_state": battery.get("batteryState"),
        "battery_soc_pct": battery.get("batterySoc"),
        "battery_capacity_ah": battery.get("batteryInstalledCapacity"),
        "battery_usable_ah": battery.get("batteryUsableCapacity"),
        "capacity_discharge_ah": battery.get("batteryCapacityToDischarge"),
        "capacity_charge_ah": battery.get("batteryCapacityToCharge"),
        "battery_voltage_v": battery.get("batteryVoltage"),
        "cell_voltage_min": battery.get("batteryMinCellVoltage"),
        "cell_voltage_max": battery.get("batteryMaxCellVoltage"),
        "battery_temp_c": battery.get("batteryAvgTemp"),
        "battery_soh_pct": battery.get("batterySoh"),
    }


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure every expected column exists so downstream tasks can rely on a stable contract.
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[EXPECTED_COLUMNS].copy()

    # Enforce types for reliable quality checks and plotting.
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Datetime conversions (UTC).
    df["last_ping_time"] = pd.to_datetime(df["last_ping_time"], utc=True, errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    # Normalize string columns.
    for col in _STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))

    return df


def _resolve_payload_column(df_raw: pd.DataFrame) -> str:
    if "payload" in df_raw.columns:
        return "payload"
    lower_map = {str(c).lower(): c for c in df_raw.columns}
    if "payload" in lower_map:
        return str(lower_map["payload"])
    raise ValueError("Input CSV must contain a 'payload' column.")


def _coerce_row_ts(parsed: Optional[Dict[str, Any]], row: pd.Series) -> Any:
    """
    Prefer epoch-ms inside parsed JSON root key `timestamp`.
    Fall back to known CSV columns if present.
    """
    if isinstance(parsed, dict):
        # Prefer root timestamp (common MQTT export shape).
        for key in ("timestamp", "ts", "created_timestamp", "createdTimestamp"):
            ts = parsed.get(key)
            if ts is not None and not (isinstance(ts, float) and pd.isna(ts)):
                return ts

        # Fallbacks in nested payload.
        payload = parsed.get("payload")
        if isinstance(payload, dict):
            for key in ("timestamp", "ts", "time", "created_timestamp", "createdTimestamp", "lastPingTime"):
                ts = payload.get(key)
                if ts is not None and not (isinstance(ts, float) and pd.isna(ts)):
                    return ts

    for col in ("timestamp", "created_timestamp", "ts"):
        if col in row.index:
            val = row[col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                return val

    return None


def parse_ev_payload(path: str) -> pd.DataFrame:
    """
    Parse raw MQTT CSV -> clean flat DataFrame.

    - Robust JSON parsing (malformed rows are skipped, but pipeline continues).
    - Missing nested objects produce nulls.
    - epoch-ms -> UTC datetime conversion.
    - Duplicate timestamps are preserved (DQ report handles dedupe metrics).
    - Returns canonical DataFrame with all 21 expected columns.
    """
    df_raw = pd.read_csv(path)
    payload_col = _resolve_payload_column(df_raw)

    records: List[Dict[str, Any]] = []
    skipped = 0

    def _parse_ts_to_utc(ts_raw: Any) -> Any:
        if ts_raw is None or (isinstance(ts_raw, float) and pd.isna(ts_raw)):
            return pd.NaT

        # Fast-path numeric epochs
        if isinstance(ts_raw, (int, float)):
            v = float(ts_raw)
            if not pd.isna(v):
                if v > 1e12:
                    return pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
                return pd.to_datetime(v, unit="s", utc=True, errors="coerce")
            return pd.NaT

        # Numeric strings: "1712345678901" or "1712345678901.0"
        if isinstance(ts_raw, str):
            s = ts_raw.strip()
            if s:
                # attempt numeric epoch first
                try:
                    v = float(s)
                    if v > 1e12:
                        return pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
                    if v > 1e9:
                        return pd.to_datetime(v, unit="s", utc=True, errors="coerce")
                except Exception:
                    pass
                # fall back to ISO / human-readable timestamps
                return pd.to_datetime(s, utc=True, errors="coerce")

        return pd.to_datetime(ts_raw, utc=True, errors="coerce")

    for _, row in df_raw.iterrows():
        parsed = safe_json_load(row.get(payload_col))
        fields = extract_fields(parsed)
        if fields is None:
            skipped += 1
            continue

        ts_raw = _coerce_row_ts(parsed, row)
        fields["ts"] = _parse_ts_to_utc(ts_raw)
        records.append(fields)

    df = pd.DataFrame.from_records(records, columns=EXPECTED_COLUMNS)
    df = _enforce_schema(df)
    df = df.sort_values(["device_id", "ts"], kind="mergesort")

    logging.info("Task1 ingestion complete: parsed=%s skipped=%s", len(df), skipped)
    return df

