import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def generate_gps_anomaly_report(df: pd.DataFrame, *, out_dir: Path = Path("outputs/task3")) -> None:
    """
    Create a Task 3 GPS anomaly HTML report (and optional PDF if WeasyPrint exists).
    Expects df to already include anomaly columns from detect_gps_anomalies().
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    d["ts"] = pd.to_datetime(d.get("ts"), utc=True, errors="coerce")
    for c in [
        "gps_lat",
        "gps_lon",
        "gps_speed_kmh",
        "delta_km_haversine",
        "delta_sec",
        "gps_anomaly_position_jump",
        "gps_anomaly_signal_dropout",
        "gps_anomaly_coordinate_freeze",
    ]:
        if c in d.columns:
            if c.startswith("gps_anomaly_"):
                d[c] = d[c].fillna(False).astype(bool)
            else:
                d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["device_id", "ts"]).sort_values(["device_id", "ts"], kind="mergesort")

    # Summary table per device
    device_counts = d.groupby("device_id", dropna=True).size().rename("n_rows")
    jump = d.groupby("device_id", dropna=True)["gps_anomaly_position_jump"].sum().rename("position_jump")
    dropout = d.groupby("device_id", dropna=True)["gps_anomaly_signal_dropout"].sum().rename("signal_dropout")
    freeze = d.groupby("device_id", dropna=True)["gps_anomaly_coordinate_freeze"].sum().rename("coordinate_freeze")
    summary = pd.concat([device_counts, jump, dropout, freeze], axis=1).fillna(0)
    for c in ["position_jump", "signal_dropout", "coordinate_freeze"]:
        summary[c] = summary[c].astype(int)
        summary[f"{c}_rate_pct"] = (summary[c] / summary["n_rows"] * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)
    summary = summary.sort_values("n_rows", ascending=False)

    # Plot: anomaly counts per device
    sns.set_style("whitegrid")
    plot_df = summary[["position_jump", "signal_dropout", "coordinate_freeze"]].reset_index().melt(
        id_vars="device_id", var_name="anomaly_type", value_name="count"
    )
    fig, ax = plt.subplots(figsize=(12, max(3.8, 0.6 + 0.5 * summary.shape[0])))
    sns.barplot(data=plot_df, x="count", y="device_id", hue="anomaly_type", ax=ax)
    ax.set_title("GPS Anomaly Counts by Device")
    ax.set_xlabel("Count")
    ax.set_ylabel("Device")
    ax.legend(loc="lower right", frameon=False)
    _save_fig(fig, fig_dir / "gps_anomaly_counts_by_device.png")

    # Plot: jump severity (distance vs time) for top events
    jumps = d[d.get("gps_anomaly_position_jump", False)].copy()
    fig_path_jump = None
    if not jumps.empty and "delta_km_haversine" in jumps.columns and "delta_sec" in jumps.columns:
        j = jumps.dropna(subset=["delta_km_haversine", "delta_sec"]).copy()
        j = j[(j["delta_sec"] > 0) & (j["delta_sec"] < 60)]
        if not j.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(j["delta_sec"], j["delta_km_haversine"], s=12, alpha=0.55)
            ax.axvline(30, color="black", linewidth=1, alpha=0.4)
            ax.axhline(1.0, color="black", linewidth=1, alpha=0.4)
            ax.set_title("Position Jumps: Δt vs Δdistance (flagged)")
            ax.set_xlabel("Δt (sec)")
            ax.set_ylabel("Δdistance (km)")
            _save_fig(fig, fig_dir / "gps_position_jump_scatter.png")
            fig_path_jump = "figures/gps_position_jump_scatter.png"

    # Styling (Poppins + leaf green/white)
    css = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <title>GPS Anomaly Report</title>
  <style>
    :root{ --leaf:#2E7D32; --white:#FFFFFF; --leaf-10:rgba(46,125,50,0.10); --leaf-18:rgba(46,125,50,0.18); }
    html,body{background:var(--white); color:var(--leaf); margin:0; padding:0;}
    body{font-family:"Poppins", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; line-height:1.55;}
    .container{max-width:1000px; margin:0 auto; padding:28px 18px 56px;}
    .card{border:1px solid var(--leaf-18); border-radius:14px; padding:18px; background:var(--white);}
    h2{font-size:22px; font-weight:700; margin:0 0 10px;}
    h3{font-size:16px; font-weight:700; margin:18px 0 10px;}
    p{margin:8px 0;}
    table.dataframe{border-collapse:separate; border-spacing:0; width:100%; margin:10px 0 0;}
    table.dataframe thead th{background:var(--leaf); color:var(--white); padding:10px 12px; text-align:left; border:1px solid var(--leaf);}
    table.dataframe tbody th{background:var(--leaf-10); padding:10px 12px; text-align:left; border:1px solid var(--leaf-18); font-weight:600;}
    table.dataframe td{background:var(--white); padding:10px 12px; text-align:left; border:1px solid var(--leaf-18);}
    img{display:block; max-width:100%; height:auto; border-radius:12px; border:1px solid var(--leaf-18); margin:10px 0;}
  </style>
</head>
<body><div class="container"><div class="card">
"""

    parts: List[str] = [css]
    parts.append("<h2>GPS Signal Quality Report (Task 3)</h2>")
    parts.append(f"<p>Total records: <b>{len(d)}</b></p>")
    parts.append(f"<p>Distinct devices: <b>{d['device_id'].nunique(dropna=True)}</b></p>")

    parts.append("<h3>Per-device anomaly summary</h3>")
    parts.append(summary.reset_index().to_html(index=False))

    parts.append("<h3>Anomaly counts by device</h3>")
    parts.append('<img src="figures/gps_anomaly_counts_by_device.png" />')

    if fig_path_jump is not None:
        parts.append("<h3>Position jump severity</h3>")
        parts.append(f'<img src="{fig_path_jump}" />')

    parts.append("</div></div></body></html>")
    html = "\n".join(parts)

    html_path = out_dir / "gps_anomaly_report.html"
    html_path.write_text(html, encoding="utf-8")

    # Optional PDF
    try:
        from weasyprint import HTML  # type: ignore

        HTML(string=html, base_url=str(out_dir.resolve())).write_pdf(str(out_dir / "gps_anomaly_report.pdf"))
    except Exception:
        pass

