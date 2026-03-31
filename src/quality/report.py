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


def generate_report(df: pd.DataFrame, *, out_dir: Path = Path("outputs/task1/reports")) -> None:
    """
    Generate a Task-1 data quality HTML report with required checks and figures.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df.get("ts"), utc=True, errors="coerce")
    df["ts"] = df["ts"]
    df = df.sort_values(["device_id", "ts"], kind="mergesort")

    canonical_cols = [c for c in df.columns if c not in {"gap_sec", "cell_imbalance"}]
    canonical_cols = [c for c in canonical_cols if c in df.columns]
    exact_dupes = int(df[canonical_cols].duplicated().sum()) if canonical_cols else int(df.duplicated().sum())

    for col in CRITICAL_NULL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Numeric coercion for range / sanity checks.
    range_cols = [
        "battery_soc_pct",
        "battery_soh_pct",
        "cell_voltage_min",
        "cell_voltage_max",
        "gps_lat",
        "gps_lon",
        "gps_speed_kmh",
        "gps_delta_km",
        "gps_total_km",
        "battery_voltage_v",
    ]
    for col in range_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gap_sec"] = df.groupby("device_id", sort=False)["ts"].diff().dt.total_seconds()
    df["cell_imbalance"] = pd.to_numeric(df.get("cell_voltage_max"), errors="coerce") - pd.to_numeric(
        df.get("cell_voltage_min"), errors="coerce"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"

    report: List[str] = []

    report.append(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <title>EV Telemetry Data Quality Report</title>
  <style>
    :root{
      --leaf:#2E7D32;
      --white:#FFFFFF;
      --leaf-10: rgba(46,125,50,0.10);
      --leaf-18: rgba(46,125,50,0.18);
      --leaf-30: rgba(46,125,50,0.30);
    }
    html,body{background:var(--white); color:var(--leaf); margin:0; padding:0;}
    body{font-family:"Poppins", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; line-height:1.55;}
    .container{max-width:1000px; margin:0 auto; padding:28px 18px 56px;}
    .card{border:1px solid var(--leaf-18); border-radius:14px; padding:18px 18px; background:var(--white);}
    h1,h2,h3,h4{margin:0 0 10px; letter-spacing:-0.02em;}
    h2{font-size:22px; font-weight:700;}
    h3{font-size:16px; font-weight:700; margin-top:18px;}
    h4{font-size:13px; font-weight:700; margin-top:14px; opacity:0.95;}
    p{margin:8px 0;}
    b,strong{font-weight:700;}
    hr{border:none; border-top:1px solid var(--leaf-18); margin:18px 0;}

    /* Tables (pandas .to_html) */
    table.dataframe{border-collapse:separate; border-spacing:0; width:100%; margin:10px 0 0;}
    table.dataframe thead th{
      position:sticky; top:0;
      background:var(--leaf);
      color:var(--white);
      font-weight:600;
      border:1px solid var(--leaf);
      padding:10px 12px;
      text-align:left;
    }
    table.dataframe tbody th{
      font-weight:600;
      border:1px solid var(--leaf-18);
      padding:10px 12px;
      text-align:left;
      background:var(--leaf-10);
    }
    table.dataframe td{
      border:1px solid var(--leaf-18);
      padding:10px 12px;
      text-align:left;
      background:var(--white);
    }
    table.dataframe tbody tr:hover td, table.dataframe tbody tr:hover th{
      background:var(--leaf-10);
    }
    img{display:block; max-width:100%; height:auto; border-radius:12px; border:1px solid var(--leaf-18); margin:10px 0;}
    .badge{display:inline-block; padding:2px 10px; border-radius:999px; border:1px solid var(--leaf-30); background:var(--leaf-10); font-weight:600; font-size:12px;}
    .muted{opacity:0.9;}
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
"""
    )

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

    # Extra sanity checks (helpful for GPS/telemetry health)
    gps_lat_bad = df[df["gps_lat"].notna() & ((df["gps_lat"] < -90) | (df["gps_lat"] > 90))]
    gps_lon_bad = df[df["gps_lon"].notna() & ((df["gps_lon"] < -180) | (df["gps_lon"] > 180))]
    gps_speed_neg = df[df["gps_speed_kmh"].notna() & (df["gps_speed_kmh"] < 0)]
    gps_delta_neg = df[df["gps_delta_km"].notna() & (df["gps_delta_km"] < 0)]
    gps_total_neg = df[df["gps_total_km"].notna() & (df["gps_total_km"] < 0)]
    pack_voltage_bad = df[df["battery_voltage_v"].notna() & (df["battery_voltage_v"] <= 0)]

    report.append("<h4>Additional sanity checks</h4>")
    report.append(f"<p>GPS latitude outside [-90,90]: <b>{len(gps_lat_bad)}</b></p>")
    report.append(f"<p>GPS longitude outside [-180,180]: <b>{len(gps_lon_bad)}</b></p>")
    report.append(f"<p>GPS speed < 0: <b>{len(gps_speed_neg)}</b></p>")
    report.append(f"<p>gps_delta_km < 0: <b>{len(gps_delta_neg)}</b></p>")
    report.append(f"<p>gps_total_km < 0: <b>{len(gps_total_neg)}</b></p>")
    report.append(f"<p>battery_voltage_v <= 0: <b>{len(pack_voltage_bad)}</b></p>")

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

    report.append("</div></div></body></html>")
    html = "\n".join(report)

    output_path = out_dir / "data_quality_report.html"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)

    root_html = html.replace('src="figures/', 'src="reports/figures/')
    (out_dir.parent / "data_quality_report.html").write_text(root_html, encoding="utf-8")

    try:
        from weasyprint import HTML  # type: ignore

        pdf_path = out_dir / "data_quality_report.pdf"
        HTML(string=html, base_url=str(out_dir.resolve())).write_pdf(str(pdf_path))

        root_pdf_path = out_dir.parent / "data_quality_report.pdf"
        HTML(string=root_html, base_url=str(out_dir.parent.resolve())).write_pdf(str(root_pdf_path))
    except Exception as e:
        import logging

        logging.warning("PDF export skipped: %s", e)