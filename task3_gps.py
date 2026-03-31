from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.gps.kalman import smooth_latlon_kalman
from src.gps.quality import detect_gps_anomalies
from src.ingestion.pipeline import parse_ev_payload
from src.trips.carbon import trips_with_carbon_credits
from src.trips.segmentation import segment_trips


def _top_devices_for_map(df: pd.DataFrame, k: int = 2) -> list:
    s = df.loc[df["gps_lat"].notna() & df["gps_lon"].notna(), "device_id"]
    if s.empty:
        return []
    vc = s.value_counts()
    return list(vc.head(k).index)


def _write_gps_map(df: pd.DataFrame, device_ids: list[str], out_html: Path) -> None:
    import folium

    sub = df[df["device_id"].isin(device_ids)].sort_values(["device_id", "ts"])
    lat_m = sub["gps_lat"].median()
    lon_m = sub["gps_lon"].median()
    if not pd.notna(lat_m) or not pd.notna(lon_m):
        lat_m = sub["gps_lat_smooth"].median()
        lon_m = sub["gps_lon_smooth"].median()
    m = folium.Map(location=[float(lat_m), float(lon_m)], zoom_start=11, tiles="OpenStreetMap")
    palette_raw = ("#1f77b4", "#ff7f0e")
    palette_smooth = ("#d62728", "#2ca02c")
    for dev, cr, cs in zip(device_ids, palette_raw, palette_smooth):
        d = sub[sub["device_id"] == dev]
        raw_pts = [
            [float(r["gps_lat"]), float(r["gps_lon"])]
            for _, r in d.iterrows()
            if pd.notna(r["gps_lat"]) and pd.notna(r["gps_lon"])
        ]
        sm_pts = [
            [float(r["gps_lat_smooth"]), float(r["gps_lon_smooth"])]
            for _, r in d.iterrows()
            if pd.notna(r["gps_lat_smooth"]) and pd.notna(r["gps_lon_smooth"])
        ]
        if len(raw_pts) >= 2:
            folium.PolyLine(raw_pts, color=cr, weight=4, opacity=0.85, tooltip=f"{dev} raw").add_to(m)
        if len(sm_pts) >= 2:
            folium.PolyLine(
                sm_pts, color=cs, weight=3, opacity=0.9, dash_array="6", tooltip=f"{dev} Kalman"
            ).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def main() -> None:
    root = Path(__file__).resolve().parent
    raw_csv = root / "data" / "raw" / "ev_prod_data.csv"
    out_dir = root / "outputs" / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_ev_payload(str(raw_csv))
    df, _masks = detect_gps_anomalies(df)
    df = smooth_latlon_kalman(df)

    devs = _top_devices_for_map(df, 2)
    if devs:
        _write_gps_map(df, devs, out_dir / "gps_map.html")
    else:
        (out_dir / "gps_map.html").write_text("<html><body>No GPS points for map.</body></html>", encoding="utf-8")

    trips = segment_trips(df)
    credits = trips_with_carbon_credits(trips)

    trips.to_csv(out_dir / "trip_segments.csv", index=False)
    credits.to_csv(out_dir / "trip_carbon_credits.csv", index=False)
    print("Task 3 completed:", out_dir)


if __name__ == "__main__":
    main()
