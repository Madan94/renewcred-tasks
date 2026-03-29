# src/ingestion/pipeline.py

import json
import pandas as pd
import hashlib
import logging
from src.config import SECRET_SALT

logging.basicConfig(level=logging.INFO)


def hash_imei(imei):
    if pd.isna(imei):
        return None
    return hashlib.sha256((imei + SECRET_SALT).encode()).hexdigest()


def safe_json_load(payload):
    try:
        return json.loads(payload)
    except Exception:
        return None


def extract_fields(data):
    if not data:
        return None
#Missing TimeSatmp Value
    payload = data.get("payload", {})

    return {
        "device_id": payload.get("deviceId"),
        "imei_token": hash_imei(payload.get("imei")),
        "last_ping_time": payload.get("lastPingTime"),
        "device_status": payload.get("status"),

        "gps_lat": payload.get("gps", {}).get("gpsLatitude"),
        "gps_lon": payload.get("gps", {}).get("gpsLongitude"),
        "gps_speed_kmh": payload.get("gps", {}).get("gpsGroundSpeed"),
        "gps_delta_km": payload.get("gps", {}).get("gpsGroundDeltaDistance"),
        "gps_total_km": payload.get("gps", {}).get("gpsTotalGroundDistance"),

        "battery_state": payload.get("battery", {}).get("batteryState"),
        "battery_soc_pct": payload.get("battery", {}).get("batterySoc"),
        "battery_capacity_ah": payload.get("battery", {}).get("batteryInstalledCapacity"),
        "battery_usable_ah": payload.get("battery", {}).get("batteryUsableCapacity"),
        "capacity_discharge_ah": payload.get("battery", {}).get("batteryCapacityToDischarge"),
        "capacity_charge_ah": payload.get("battery", {}).get("batteryCapacityToCharge"),
        "battery_voltage_v": payload.get("battery", {}).get("batteryVoltage"),
        "cell_voltage_min": payload.get("battery", {}).get("batteryMinCellVoltage"),
        "cell_voltage_max": payload.get("battery", {}).get("batteryMaxCellVoltage"),
        "battery_temp_c": payload.get("battery", {}).get("batteryAvgTemp"),
        "battery_soh_pct": payload.get("battery", {}).get("batterySoh"),
    }


def parse_ev_payload(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path)

    records = []

    for _, row in df_raw.iterrows():
        parsed = safe_json_load(row.get("payload"))
        fields = extract_fields(parsed)

        if not fields:
            continue

        try:
            ts = pd.to_datetime(row.get("timestamp"), unit="ms", utc=True)
        except Exception:
            ts = None

        fields["ts"] = ts
        records.append(fields)

    df = pd.DataFrame(records)

    # Cleanups
    df = df.drop_duplicates()
    df = df.sort_values(["device_id", "ts"])

    logging.info(f"Final parsed rows: {len(df)}")

    return df

