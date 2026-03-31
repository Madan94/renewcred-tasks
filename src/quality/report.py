import matplotlib

matplotlib.use("Agg")

import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import CRITICAL_NULL_COLUMNS


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.clf()


def _device_gap_histograms(df: pd.DataFrame, figures_dir: Path, max_devices_per_fig: int = 9) -> List[str]:
    """
    Plot inter-ping gap distributions per device.
    Returns relative html img src paths.
    """
    devices = df["device_id"].dropna().unique().tolist()
    if not devices:
        return []

    # Keep stable ordering.
    devices = sorted(devices, key=lambda d: (df["device_id"] == d).sum(), reverse=True)

    chunk_size = max_devices_per_fig
    parts: List[str] = []
    gap_col = "gap_sec"

    total_devices = len(devices)
    n_parts = max(1, math.ceil(total_devices / chunk_size))

    sns.set_style("whitegrid")

    for part_idx in range(n_parts):
        start = part_idx * chunk_size
        end = min(total_devices, start + chunk_size)
        chunk = devices[start:end]

        cols = 3
        rows = math.ceil(len(chunk) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows))
        axes_arr = np.array(axes).reshape(-1)

        for ax_i, device_id in enumerate(chunk):
            ax = axes_arr[ax_i]
            gaps = df.loc[df["device_id"] == device_id, gap_col].dropna().values
            if gaps.size == 0:
                ax.set_title(str(device_id)[:10] + " (no gaps)")
                ax.axis("off")
                continue

            # Cap extreme gaps for readability.
            cap = np.nanpercentile(gaps, 99.0)
            cap = max(float(cap), 1.0)
            gaps = gaps[gaps <= cap]
            ax.hist(gaps, bins=40)
            ax.set_title(str(device_id)[:10])
            ax.set_xlabel("Inter-ping gap (sec)")
            ax.set_ylabel("Count")

        # Turn off unused axes.
        for ax_i in range(len(chunk), len(axes_arr)):
            axes_arr[ax_i].axis("off")

        fname = f"inter_ping_gap_distributions_part{part_idx + 1}.png"
        out_path = figures_dir / fname
        _save_fig(fig, out_path)
        parts.append(f"figures/{fname}")

    return parts


def generate_report(df: pd.DataFrame) -> None:
    """
    Generate a Task-1 data quality HTML report with required checks and figures.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df.get("ts"), utc=True, errors="coerce")
    df["ts"] = df["ts"]
    df = df.sort_values(["device_id", "ts"], kind="mergesort")

    # Derived / normalized columns used by multiple checks.
    for col in CRITICAL_NULL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Numeric coercion for range checks.
    range_cols = ["battery_soc_pct", "battery_soh_pct", "cell_voltage_min", "cell_voltage_max"]
    for col in range_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gap_sec"] = df.groupby("device_id", sort=False)["ts"].diff().dt.total_seconds()
    df["cell_imbalance"] = pd.to_numeric(df.get("cell_voltage_max"), errors="coerce") - pd.to_numeric(
        df.get("cell_voltage_min"), errors="coerce"
    )

    figures_dir = Path("outputs/reports/figures")

    report: List[str] = []

    report.append("<h2>EV Telemetry Data Quality Report (Task 1)</h2>")
    report.append(f"<p>Total records: <b>{len(df)}</b></p>")
    report.append(f"<p>Distinct devices: <b>{df['device_id'].nunique(dropna=True)}</b></p>")

    # Null Rates
    null_rates = (df.isna().mean() * 100).round(2)
    report.append("<h3>Null Rates (%)</h3>")
    report.append(null_rates.to_frame().to_html())

    report.append("<h4>Signal columns with >5% nulls</h4>")
    flagged = []
    for col in CRITICAL_NULL_COLUMNS:
        if col in null_rates.index and null_rates[col] > 5:
            flagged.append(f"{col}: {null_rates[col]}%")
    report.append("<p>" + (", ".join(flagged) if flagged else "None") + "</p>")

    # Duplicates
    exact_dupes = int(df.duplicated().sum())
    report.append(f"<h3>Duplicates</h3>")
    report.append(f"<p>Exact duplicate rows (full-row match): <b>{exact_dupes}</b></p>")

    near_dupe_mask = df["gap_sec"].notna() & (df["gap_sec"] <= 1.0)
    near_dupes = int(near_dupe_mask.sum())
    report.append(
        f"<p>Same device + ts within 1s (<=1.0s from previous ping per device): <b>{near_dupes}</b></p>"
    )

    # Out-of-range
    soc_issues = df[df["battery_soc_pct"].notna() & ((df["battery_soc_pct"] < 0) | (df["battery_soc_pct"] > 100))]
    soh_issues = df[df["battery_soh_pct"].notna() & ((df["battery_soh_pct"] < 0) | (df["battery_soh_pct"] > 100))]
    voltage_issues = df[
        (df["cell_voltage_min"].notna() & (df["cell_voltage_min"] < 2.5)) |
        (df["cell_voltage_max"].notna() & (df["cell_voltage_max"] > 4.2))
    ]
    report.append("<h3>Out-of-range flags</h3>")
    report.append(f"<p>SoC outside [0,100]: <b>{len(soc_issues)}</b></p>")
    report.append(f"<p>SoH outside [0,100]: <b>{len(soh_issues)}</b></p>")
    report.append(f"<p>Cell voltage outside [2.5V,4.2V]: <b>{len(voltage_issues)}</b></p>")

    # Temporal continuity + delayed pings
    delay_mask = df["gap_sec"].notna() & (df["gap_sec"] > 60)
    device_totals = df.groupby("device_id", dropna=True).size()
    device_delays = df.loc[delay_mask].groupby("device_id", dropna=True).size()
    delay_pct = (device_delays / device_totals).fillna(0).sort_values(ascending=False)

    report.append("<h3>Temporal continuity</h3>")
    report.append("<h4>Inter-ping gap distribution (per device)</h4>")
    gap_imgs = _device_gap_histograms(df, figures_dir)
    if gap_imgs:
        for img_src in gap_imgs:
            report.append(f'<img src="{img_src}" style="max-width: 100%; height: auto;" />')

    report.append("<h4>Devices with >10% delayed pings (>60s)</h4>")
    delayed_devices = delay_pct[delay_pct > 0.1]
    if delayed_devices.empty:
        report.append("<p>None</p>")
    else:
        report.append(delayed_devices.to_frame(name="delayed_ratio").to_html())

    # Cell imbalance histogram
    report.append("<h3>Cell voltage imbalance</h3>")
    imbalance = df["cell_imbalance"].dropna()
    if len(imbalance) > 0:
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(9, 5))
        cap = float(np.nanpercentile(imbalance.values, 99.0)) if len(imbalance) else 1.0
        cap = max(cap, 0.01)
        imbalance_plot = imbalance[imbalance <= cap]
        ax.hist(imbalance_plot.values, bins=50)
        ax.set_title("Cell Imbalance Histogram (cell_voltage_max - cell_voltage_min)")
        ax.set_xlabel("Imbalance (V)")
        ax.set_ylabel("Count")
        _save_fig(fig, figures_dir / "cell_imbalance_histogram.png")
        report.append('<img src="figures/cell_imbalance_histogram.png" style="max-width: 100%; height: auto;" />')
    else:
        report.append("<p>No cell imbalance values available (insufficient voltage data).</p>")

    # Save HTML
    html = "\n".join(report)

    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data_quality_report.html"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)

    # Also write a repo-root copy to match the assignment deliverable name.
    # The only difference is the relative paths for embedded figures.
    root_html = html.replace('src="figures/', 'src="outputs/reports/figures/')
    with Path("data_quality_report.html").open("w", encoding="utf-8") as f:
        f.write(root_html)