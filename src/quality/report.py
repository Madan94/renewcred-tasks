import pandas as pd
from src.config import *


def generate_report(df: pd.DataFrame):

    report = []

    # Null Rates
    null_rates = (df.isnull().mean() * 100).round(2)
    report.append("<h2>Null Rates (%)</h2>")
    report.append(null_rates.to_frame().to_html())

    # Critical null flags
    report.append("<h3>Critical Column Null Alerts</h3>")
    for col in CRITICAL_NULL_COLUMNS:
        if null_rates[col] > 5:
            report.append(f"<p>{col} has {null_rates[col]}% nulls</p>")

    # Duplicates
    exact_dupes = df.duplicated().sum()
    report.append(f"<h2>Exact Duplicates: {exact_dupes}</h2>")

    df["ts_diff"] = df.groupby("device_id")["ts"].diff().dt.total_seconds()
    near_dupes = (df["ts_diff"] < 1).sum()
    report.append(f"<h2>Near Duplicates (<1 sec): {near_dupes}</h2>")

    # Out-of-range
    soc_issues = df[(df["battery_soc_pct"] < 0) | (df["battery_soc_pct"] > 100)]
    report.append(f"<h2>SoC Out-of-Range: {len(soc_issues)}</h2>")

    soh_issues = df[(df["battery_soh_pct"] < 0) | (df["battery_soh_pct"] > 100)]
    report.append(f"<h2>SoH Out-of-Range: {len(soh_issues)}</h2>")

    voltage_issues = df[
        (df["cell_voltage_min"] < 2.5) |
        (df["cell_voltage_max"] > 4.2)
    ]
    report.append(f"<h2>Voltage Out-of-Range: {len(voltage_issues)}</h2>")

    # Temporal gaps
    df["gap_sec"] = df.groupby("device_id")["ts"].diff().dt.total_seconds()
    delayed = df[df["gap_sec"] > 60]

    delay_pct = (delayed.groupby("device_id").size() /
                 df.groupby("device_id").size()).fillna(0)

    report.append("<h2>Devices with >10% delayed pings</h2>")
    report.append(delay_pct[delay_pct > 0.1].to_frame().to_html())

    # Cell imbalance
    df["cell_imbalance"] = df["cell_voltage_max"] - df["cell_voltage_min"]
    report.append("<h2>Cell Imbalance Stats</h2>")
    report.append(df["cell_imbalance"].describe().to_frame().to_html())

    # Save HTML
    html = "".join(report)
    with open("outputs/reports/data_quality_report.html", "w") as f:
        f.write(html)