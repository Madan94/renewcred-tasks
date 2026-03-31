import matplotlib

matplotlib.use("Agg")

import math
from pathlib import Path
from typing import List, Sequence

import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _top_devices_by_count(df: pd.DataFrame, n: int) -> List[str]:
    if "device_id" not in df.columns:
        return []
    counts = df["device_id"].value_counts(dropna=True)
    return counts.head(n).index.astype(str).tolist()


def generate_all_charts(df: pd.DataFrame, *, output_dir: Path = Path("outputs/task1/eda_charts")) -> None:
    df = df.copy()
    df["ts"] = pd.to_datetime(df.get("ts"), utc=True, errors="coerce")

    _ensure_output_dir(output_dir)

    # 1. SoC distribution across all devices
    soc = pd.to_numeric(df.get("battery_soc_pct"), errors="coerce")
    soc = soc.dropna()
    plt.figure(figsize=(10, 5))
    if len(soc) > 0:
        sns.set_style("whitegrid")
        sns.violinplot(y=soc, color="#4C72B0", inner=None, cut=0)
        q1, med, q3 = soc.quantile([0.25, 0.5, 0.75]).tolist()
        # IQR + median marker
        plt.plot([0], [med], marker="o", color="black", markersize=6, zorder=5)
        plt.vlines(0, q1, q3, color="black", linewidth=3, zorder=4)
        plt.text(
            0.06,
            med,
            f"median={med:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            color="black",
        )
    plt.title("SoC Distribution Across Fleet")
    plt.ylabel("SoC (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "soc_distribution.png", dpi=160)
    plt.close()

    # 2. Battery temperature vs SoC (colored by battery_state)
    plt.figure(figsize=(10, 5))
    temp_soc_df = df[["battery_soc_pct", "battery_temp_c", "battery_state"]].copy()
    temp_soc_df["battery_soc_pct"] = pd.to_numeric(temp_soc_df["battery_soc_pct"], errors="coerce")
    temp_soc_df["battery_temp_c"] = pd.to_numeric(temp_soc_df["battery_temp_c"], errors="coerce")
    temp_soc_df["battery_state"] = temp_soc_df["battery_state"].fillna("Unknown").astype(str)
    temp_soc_df = temp_soc_df.dropna(subset=["battery_soc_pct", "battery_temp_c"])
    if not temp_soc_df.empty:
        sns.scatterplot(
            data=temp_soc_df,
            x="battery_soc_pct",
            y="battery_temp_c",
            hue="battery_state",
            s=14,
            alpha=0.65,
        )
        # Overall trend line (no extra color palette; keeps chart readable)
        sns.regplot(
            data=temp_soc_df,
            x="battery_soc_pct",
            y="battery_temp_c",
            scatter=False,
            color="black",
            line_kws={"linewidth": 2, "alpha": 0.55},
        )
        plt.legend(loc="best", fontsize="small", frameon=False, title="battery_state")
    plt.title("Battery Temperature vs SoC")
    plt.tight_layout()
    plt.savefig(output_dir / "temp_vs_soc.png", dpi=160)
    plt.close()

    # 3. Per-device SoH over time (multi-line)
    plt.figure(figsize=(12, 6))
    soh_col = "battery_soh_pct"
    df[soh_col] = pd.to_numeric(df.get(soh_col), errors="coerce")
    devices = _top_devices_by_count(df, n=5)
    if len(devices) == 0 and df["device_id"].notna().any():
        devices = df["device_id"].dropna().astype(str).unique().tolist()[:5]
    for d in devices:
        sub = df[df["device_id"].astype(str) == d].sort_values("ts")
        if sub.empty:
            continue
        y = sub[soh_col].dropna()
        delta = float(y.iloc[-1] - y.iloc[0]) if len(y) >= 2 else float("nan")
        label = f"{d[:10]}" + (f" (Δ{delta:+.2f})" if np.isfinite(delta) else "")
        plt.plot(sub["ts"], sub[soh_col], linewidth=1.5, label=label)
    if devices:
        plt.legend(loc="best", fontsize="small", ncol=2, frameon=False)
    plt.title("Battery SoH Over Time (Sample Devices)")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("SoH (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "soh_timeseries.png", dpi=160)
    plt.close()

    # 4. GPS heatmap of vehicle locations
    gps_df = df[["gps_lat", "gps_lon"]].copy()
    gps_df["gps_lat"] = pd.to_numeric(gps_df["gps_lat"], errors="coerce")
    gps_df["gps_lon"] = pd.to_numeric(gps_df["gps_lon"], errors="coerce")
    gps_df = gps_df.dropna(subset=["gps_lat", "gps_lon"])
    gps_df = gps_df[(gps_df["gps_lat"] != 0) & (gps_df["gps_lon"] != 0)]

    if not gps_df.empty:
        center_lat = float(gps_df["gps_lat"].mean())
        center_lon = float(gps_df["gps_lon"].mean())
    else:
        center_lat, center_lon = 20.0, 78.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    heat_data = gps_df[["gps_lat", "gps_lon"]].values.tolist()
    if heat_data:
        HeatMap(heat_data).add_to(m)
        # Top "hot spots" (simple grid clustering to add interpretability)
        grid = gps_df.copy()
        grid["lat_r"] = grid["gps_lat"].round(2)
        grid["lon_r"] = grid["gps_lon"].round(2)
        hot = (
            grid.groupby(["lat_r", "lon_r"], dropna=True)
            .size()
            .sort_values(ascending=False)
            .head(3)
        )
        for (lat_r, lon_r), cnt in hot.items():
            folium.CircleMarker(
                location=[float(lat_r), float(lon_r)],
                radius=8,
                weight=2,
                color="#1B5E20",
                fill=True,
                fill_opacity=0.25,
                popup=f"~{int(cnt)} points near ({lat_r}, {lon_r})",
            ).add_to(m)
    m.save(str(output_dir / "gps_heatmap.html"))

    # 5. Correlation heatmap: battery signals x gps signals
    battery_cols = [
        "battery_soc_pct",
        "battery_temp_c",
        "battery_voltage_v",
        "battery_soh_pct",
        "cell_voltage_min",
        "cell_voltage_max",
        "battery_capacity_ah",
        "battery_usable_ah",
        "capacity_discharge_ah",
        "capacity_charge_ah",
    ]
    gps_cols = ["gps_lat", "gps_lon", "gps_speed_kmh", "gps_delta_km", "gps_total_km"]

    battery_cols = [c for c in battery_cols if c in df.columns]
    gps_cols = [c for c in gps_cols if c in df.columns]

    corr_out_path = output_dir / "correlation_heatmap.png"
    plt.figure(figsize=(10, 6))
    if battery_cols and gps_cols:
        corr_df = df[battery_cols + gps_cols].apply(pd.to_numeric, errors="coerce")
        corr = corr_df.corr(numeric_only=True)
        corr_matrix = corr.loc[battery_cols, gps_cols]
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 8},
            cbar_kws={"shrink": 0.9},
        )
    else:
        plt.text(0.5, 0.5, "Insufficient columns for correlation heatmap", ha="center")
    plt.title("Correlation Heatmap: Battery x GPS")
    plt.tight_layout()
    plt.savefig(corr_out_path, dpi=160)
    plt.close()

    # 6. Operational state timeline: Gantt-style chart for 3 devices
    plt.figure(figsize=(12, 1.8 * 3 + 2.5))
    sample_devices = _top_devices_by_count(df, n=3)
    if len(sample_devices) < 3 and df["device_id"].notna().any():
        more = df["device_id"].dropna().astype(str).unique().tolist()
        sample_devices = (sample_devices + more)[:3]

    state_colors = {
        "Charging": "#F28E2B",
        "Active": "#59A14F",
        "Inactive": "#B0B0B0",
    }

    fig, ax = plt.subplots(figsize=(12, 1.8 * max(1, len(sample_devices)) + 3))
    ax.set_title("Operational State Timeline (Gantt-style, 3 Sample Devices)")

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    for row_idx, device_id in enumerate(sample_devices):
        sub = df[df["device_id"].astype(str) == str(device_id)].sort_values("ts").copy()
        if sub.empty:
            continue

        # Charging overrides Active/Inactive for this timeline.
        battery_state = sub["battery_state"].fillna("").astype(str)
        device_status = sub["device_status"].fillna("Inactive").astype(str)
        op_state = device_status.where(battery_state.ne("Charging"), "Charging")
        sub["op_state"] = op_state

        # Build contiguous segments.
        change_id = sub["op_state"].ne(sub["op_state"].shift()).cumsum()
        segments = sub.groupby(change_id).agg(
            state=("op_state", "first"),
            start_ts=("ts", "min"),
            end_ts=("ts", "max"),
        )
        segments = segments.dropna(subset=["start_ts", "end_ts"])
        if segments.empty:
            continue

        # Percent time in each state (approx by segment durations).
        seg_dur_s = (segments["end_ts"] - segments["start_ts"]).dt.total_seconds().clip(lower=0)
        total_s = float(seg_dur_s.sum()) if len(seg_dur_s) else 0.0
        pct = {}
        if total_s > 0:
            for st in state_colors.keys():
                m_st = segments["state"] == st
                pct[st] = float(seg_dur_s.loc[m_st].sum()) / total_s * 100.0
        pct_txt = "  ".join([f"{st[0]}:{pct.get(st, 0.0):.0f}%" for st in state_colors.keys()])

        y_tuple = (row_idx - 0.4, 0.8)
        for state, color in state_colors.items():
            seg_state = segments[segments["state"] == state]
            if seg_state.empty:
                continue
            start_nums = mdates.date2num(np.array(seg_state["start_ts"].dt.to_pydatetime()))
            end_nums = mdates.date2num(np.array(seg_state["end_ts"].dt.to_pydatetime()))
            widths = np.maximum(end_nums - start_nums, 1e-6)
            ax.broken_barh(list(zip(start_nums, widths)), y_tuple, facecolors=color, edgecolor="black", linewidth=0.2)

        # Add state mix annotation to the right of each lane.
        x_right = ax.get_xlim()[1]
        ax.text(x_right, row_idx, pct_txt, va="center", ha="left", fontsize=9, color="black")

    ax.set_yticks(range(len(sample_devices)))
    ax.set_yticklabels([str(d)[:10] for d in sample_devices])
    ax.set_xlabel("Timestamp (UTC)")
    ax.margins(x=0.12)

    legend_handles = []
    import matplotlib.patches as mpatches

    for state, color in state_colors.items():
        legend_handles.append(mpatches.Patch(color=color, label=state))
    ax.legend(handles=legend_handles, loc="upper right", ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "operational_state_timeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)