# Renewcred Tasks — EV Telemetry Pipeline (Tasks 1–3)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)

End-to-end, production-style pipelines on raw EV MQTT telemetry:

- **Task 1**: ingestion → canonical schema → data quality report + EDA charts
- **Task 2**: SoC \(t+10min\) feature engineering → **XGBoost** + **LSTM** → SHAP + anomaly flags
- **Task 3**: GPS quality (anomalies + Kalman smoothing) → map overlay → trip segmentation → carbon credits

Minimal dependencies, deterministic outputs, and artifact-first workflows.

---

## Quickstart

### Requirements

- **Python**: 3.10+
- **Input data**: place raw file at `data/raw/ev_prod_data.csv`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tasks:

```bash
PYTHONPATH=. python task1_pipeline.py
PYTHONPATH=. python task2_soc_model.py
PYTHONPATH=. python task3_gps.py
```

Notebook runners (mirrors the scripts):

```bash
jupyter lab
```

- `task1_pipeline.ipynb`
- `task2_soc_model.ipynb`
- `task3_gps.ipynb`

---

## Repository layout

```text
.
├── data/
│   └── raw/
│       └── ev_prod_data.csv
├── src/
│   ├── ingestion/
│   │   └── pipeline.py
│   ├── quality/
│   │   └── report.py
│   ├── eda/
│   │   └── visualizations.py
│   ├── features/
│   │   └── soc_features.py
│   ├── models/
│   │   ├── xgb_soc.py
│   │   └── lstm_soc.py
│   ├── anomaly/
│   │   └── bms_anomaly.py
│   ├── gps/
│   │   ├── quality.py
│   │   └── kalman.py
│   └── trips/
│       ├── segmentation.py
│       └── carbon.py
├── outputs/
│   ├── reports/
│   │   ├── data_quality_report.html
│   │   └── figures/
│   ├── task2/
│   └── task3/
├── models/
├── task1_pipeline.py
├── task1_pipeline.ipynb
├── task2_soc_model.py
├── task3_gps.py
├── task2_soc_model.ipynb
├── task3_gps.ipynb
└── requirements.txt
```

---

## Data contract

### Input format

Raw input is an MQTT export CSV with at least:

- **`payload`**: JSON string containing nested telemetry (required)
- **`timestamp`**: optional; if missing, `payload.timestamp` is used

### Canonical schema (Task 1 output frame)

`src/ingestion/pipeline.py:parse_ev_payload()` returns a flat DataFrame with **21 columns**:

| Column | Type | Notes |
|---|---:|---|
| `device_id` | string | anonymized stable ID |
| `imei_token` | string | salted hash token derived from IMEI |
| `last_ping_time` | string/nullable | raw payload field when present |
| `device_status` | string | e.g., `Active` / `Inactive` |
| `gps_lat` | float | latitude |
| `gps_lon` | float | longitude |
| `gps_speed_kmh` | float | speed (km/h) |
| `gps_delta_km` | float | delta distance between pings (km) |
| `gps_total_km` | float | odometer-like total distance (km) |
| `battery_state` | string | e.g., `Charging` / `Discharging` |
| `battery_soc_pct` | float | SoC % |
| `battery_capacity_ah` | float | nominal capacity |
| `battery_usable_ah` | float | usable capacity |
| `capacity_discharge_ah` | float | discharge counter |
| `capacity_charge_ah` | float | charge counter |
| `battery_voltage_v` | float | pack voltage |
| `cell_voltage_min` | float | min cell voltage |
| `cell_voltage_max` | float | max cell voltage |
| `battery_temp_c` | float | temperature |
| `battery_soh_pct` | float | SoH % |
| `ts` | datetime (UTC) | canonical timestamp |

---

## End-to-end workflow

### Task 1 — Ingestion → Data Quality → EDA

**Entry point**: `task1_pipeline.py`

Pipeline:

```text
raw MQTT CSV
  └─ parse_ev_payload()  -> canonical 21-col frame (sorted by device_id, ts; ts is UTC)
      ├─ generate_report()     -> HTML data quality report (+ PNG figures)
      └─ generate_all_charts() -> EDA charts (PNG + HTML map)
```

**Data Quality checks** (`src/quality/report.py`):

- **Null rates** per column (flags critical signals with >5% nulls)
- **Duplicates**
  - exact duplicate rows
  - near-duplicates: per-device inter-ping gap \(\le 1s\)
- **Out-of-range**
  - SoC outside \([0, 100]\)
  - SoH outside \([0, 100]\)
  - cell voltages outside \([2.5V, 4.2V]\)
- **Temporal continuity**
  - inter-ping gap histograms per device
  - delayed ping ratio: gap \(> 60s\) (flags devices where ratio > 10%)
- **Cell imbalance**
  - histogram of `(cell_voltage_max - cell_voltage_min)`

**EDA artifacts** (`src/eda/visualizations.py`):

- SoC distribution (violin)
- Temperature vs SoC (scatter, colored by battery_state)
- SoH time series (top devices)
- GPS heatmap (`folium`)
- Correlation heatmap (battery × GPS)
- Operational state timeline (Gantt-style)

---

### Task 2 — SoC prediction \(t+10min\) + anomaly flags

**Entry point**: `task2_soc_model.py`

Pipeline:

```text
canonical frame
  └─ build_soc_dataset(horizon=10min, cadence=30s)
      ├─ time_based_split_per_device(test_ratio=0.2)
      ├─ Model A: train_xgboost_soc(TimeSeriesSplit) + SHAP
      ├─ Model B: train_lstm_soc(seq_len=10, early stopping)
      └─ build_anomaly_flags_csv() -> anomaly_flags.csv
```

#### Feature engineering

Implemented in `src/features/soc_features.py`. Target is:

- **`target_soc_t_plus_10min`**: SoC shifted by 20 steps at 30s cadence

Feature list:

| Feature | Description |
|---|---|
| `soc_delta_5min` | SoC delta over 5 minutes |
| `discharge_rate_wh` | ΔSoC × usable_ah × voltage × 10 |
| `cell_imbalance` | cell_voltage_max − cell_voltage_min |
| `temp_deviation` | temp − rolling 7D mean temp |
| `soh_adjusted_cap` | usable_ah × (soh/100) |
| `charge_headroom` | capacity_charge_ah / battery_usable_ah |
| `speed_x_soc` | gps_speed_kmh × SoC |
| `rolling_soc_std_1h` | 1-hour rolling std of SoC |
| `idle_energy_drain` | discharge proxy only when speed == 0 |
| `rolling_speed_mean_5min` | 5-minute rolling mean speed |
| `rolling_temp_std_1h` | 1-hour rolling std temp |

#### Models

- **XGBoost** (`src/models/xgb_soc.py`)
  - `TimeSeriesSplit` CV over training set
  - exports metrics to JSON and SHAP summary plot
- **LSTM** (`src/models/lstm_soc.py`)
  - 2-layer LSTM + dropout 0.2
  - `seq_len=10`
  - early stopping by validation loss
  - median imputation (train) → `StandardScaler`

#### Model comparison (generated)

`outputs/task2/model_comparison_table.csv`:

| Metric | XGBoost | LSTM | Notes |
|---|---:|---:|---|
| Test RMSE (SoC %) | 6.7613 | 17.7554 | lower is better |
| Test MAE (SoC %) | 0.7503 | 6.8565 | lower is better |
| Training time (sec) | 3.22 | 63.7 | |
| Inference latency (ms/sample) | 0.0025 | 0.0112 | |
| Interpretability (1–5) | 4 | 2 | |
| Production readiness (1–5) | 5 | 3 | |

#### Anomaly flags

Generated by `src/anomaly/bms_anomaly.py:build_anomaly_flags_csv()` as:

| Column | Type | Description |
|---|---:|---|
| `device_id` | string | device identifier |
| `ts` | datetime (UTC) | timestamp |
| `anomaly_type` | string | rule name |
| `confidence_score` | float | IsolationForest-based confidence \(0..1\) |
| `carbon_credit_impact_pct` | float | downstream impact proxy \(0..100\) |

Anomaly types:

- `cell_imbalance_high`
- `soh_drop_7d`
- `discharge_spike_vs_rolling`
- `temperature_spike`

---

### Task 3 — GPS quality + smoothing + trips + carbon credits

**Entry point**: `task3_gps.py`

Pipeline:

```text
canonical frame
  ├─ detect_gps_anomalies()
  │    ├─ position jump     (Δdistance > 1km in <30s)
  │    ├─ signal dropout    (lat==0 and lon==0)
  │    └─ coordinate freeze (same lat/lon for >5 pings while speed>2)
  ├─ smooth_latlon_kalman() -> gps_lat_smooth, gps_lon_smooth
  ├─ gps_map.html (Folium): raw vs smoothed for top 2 devices
  ├─ segment_trips()        -> trip_segments.csv
  └─ trips_with_carbon_credits() -> trip_carbon_credits.csv
```

#### GPS anomaly detection

Implemented in `src/gps/quality.py`:

| Flag column | Condition |
|---|---|
| `gps_anomaly_position_jump` | Haversine Δdistance > 1km AND Δt < 30s |
| `gps_anomaly_signal_dropout` | `gps_lat == 0.0` AND `gps_lon == 0.0` |
| `gps_anomaly_coordinate_freeze` | same lat/lon for **> 5** consecutive pings while `gps_speed_kmh > 2` |

#### Kalman smoothing

Implemented in `src/gps/kalman.py`:

- per-device, independent 1D random-walk Kalman filter for `lat` and `lon`
- dropouts (NaN) skip measurement update; state continues smoothly

#### Trip segmentation

Implemented in `src/trips/segmentation.py`:

| Rule | Condition |
|---|---|
| Trip start | `gps_speed_kmh > 2` AND `battery_state == 'Discharging'` AND `device_status == 'Active'` |
| Trip end | `battery_state == 'Charging'` OR sustained `gps_speed_kmh == 0` for >3 min |
| Minimum distance | sum(`gps_delta_km`) >= 0.1 km |

Trip outputs:

| Column | Description |
|---|---|
| `trip_id` | `{device_id}_{trip_num}` |
| `start_ts`, `end_ts` | UTC timestamps |
| `distance_km` | sum of `gps_delta_km` |
| `avg_speed_kmh` | mean of `gps_speed_kmh` |
| `energy_consumed_wh` | \(\Delta SoC\) × usable_ah × voltage × 10 |
| `start_lat/lon`, `end_lat/lon` | first/last non-null gps points |

#### Carbon credits

Implemented in `src/trips/carbon.py`:

\[
net\_tco2e = (energy\_wh/1000)\times 0.716 \;-\; distance\_km \times 0.000150
\]

Adds a `net_tco2e` column to the per-trip table.

---

## Outputs (artifacts)

### Task 1

| Artifact | Path |
|---|---|
| Data quality report (HTML) | `outputs/reports/data_quality_report.html` |
| Report figures | `outputs/reports/figures/` |
| EDA charts (PNG/HTML) | `eda_charts/` |
| Repo-root copy of report | `data_quality_report.html` |

### Task 2

| Artifact | Path |
|---|---|
| Feature dataset | `outputs/task2/soc_features.csv` |
| XGBoost model | `models/xgboost_soc.pkl` |
| XGBoost metrics | `outputs/task2/xgboost_metrics.json` |
| SHAP summary | `outputs/task2/shap_xgboost_summary.png` |
| LSTM model bundle | `models/lstm_soc.pt` |
| LSTM metrics | `outputs/task2/lstm_metrics.json` |
| Model comparison table | `outputs/task2/model_comparison_table.csv` |
| Anomaly flags (copy) | `outputs/task2/anomaly_flags.csv` |
| Anomaly flags (root copy) | `anomaly_flags.csv` |

### Task 3

| Artifact | Path |
|---|---|
| GPS map (raw vs smoothed) | `outputs/task3/gps_map.html` |
| Trip segments | `outputs/task3/trip_segments.csv` |
| Trip carbon credits | `outputs/task3/trip_carbon_credits.csv` |

---

## Reproducibility and conventions

- **Timestamps**: canonical `ts` is always parsed/coerced to **UTC**
- **Sorting**: pipelines consistently sort by **`device_id`, `ts`** for deterministic results
- **Robust parsing**: malformed JSON rows are skipped; schema is enforced
- **Model safety**: XGBoost CV is configured with `n_jobs=1` to avoid loky/worker failures on constrained machines

---

## Security & privacy

- Device identity uses a salted hash token (`imei_token`) derived from IMEI.
- Secret salt lives in `src/config.py` (`SECRET_SALT`). In real deployments, load this from environment/secret manager instead of source.

---

## License

MIT License (see `LICENSE` if present).

# renewcred-tasks
RenewCred Intern Taks
