from __future__ import annotations

from pathlib import Path

import pandas as pd


def top_devices_for_map(df: pd.DataFrame, k: int = 2) -> list[str]:
    s = df.loc[df["gps_lat"].notna() & df["gps_lon"].notna(), "device_id"]
    if s.empty:
        return []
    return list(s.value_counts().head(k).index.astype(str))


def write_raw_vs_smoothed_map(
    df: pd.DataFrame,
    *,
    device_ids: list[str],
    out_html: Path,
    tiles: str = "OpenStreetMap",
) -> None:
    import folium

    sub = df[df["device_id"].isin(device_ids)].sort_values(["device_id", "ts"], kind="mergesort")
    lat_m = sub["gps_lat"].median()
    lon_m = sub["gps_lon"].median()
    if not pd.notna(lat_m) or not pd.notna(lon_m):
        lat_m = sub.get("gps_lat_smooth").median()
        lon_m = sub.get("gps_lon_smooth").median()
    if not pd.notna(lat_m) or not pd.notna(lon_m):
        lat_m, lon_m = 20.0, 78.0

    m = folium.Map(location=[float(lat_m), float(lon_m)], zoom_start=11, tiles=tiles)
    palette_raw = ("#1f77b4", "#ff7f0e")
    palette_smooth = ("#d62728", "#2ca02c")
    all_pts: list[list[float]] = []

    for dev, cr, cs in zip(device_ids, palette_raw, palette_smooth):
        d = sub[sub["device_id"].astype(str) == str(dev)]
        raw_pts = [
            [float(r["gps_lat"]), float(r["gps_lon"])]
            for _, r in d.iterrows()
            if pd.notna(r.get("gps_lat")) and pd.notna(r.get("gps_lon"))
        ]
        sm_pts = [
            [float(r["gps_lat_smooth"]), float(r["gps_lon_smooth"])]
            for _, r in d.iterrows()
            if pd.notna(r.get("gps_lat_smooth")) and pd.notna(r.get("gps_lon_smooth"))
        ]

        if len(raw_pts) >= 2:
            folium.PolyLine(raw_pts, color=cr, weight=4, opacity=0.85, tooltip=f"{dev} raw").add_to(m)
            all_pts.extend(raw_pts)
        if len(sm_pts) >= 2:
            folium.PolyLine(
                sm_pts,
                color=cs,
                weight=3,
                opacity=0.9,
                dash_array="6",
                tooltip=f"{dev} Kalman",
            ).add_to(m)
            all_pts.extend(sm_pts)

    if len(all_pts) >= 2:
        lats = [p[0] for p in all_pts]
        lons = [p[1] for p in all_pts]
        sw = [float(min(lats)), float(min(lons))]
        ne = [float(max(lats)), float(max(lons))]
        m.fit_bounds([sw, ne], padding=(18, 18))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))

