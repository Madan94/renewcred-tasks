from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.gps.kalman import smooth_latlon_kalman
from src.gps.quality import detect_gps_anomalies
from src.gps.compare_map import top_devices_for_map, write_raw_vs_smoothed_map
from src.gps.report import generate_gps_anomaly_report
from src.gps.map_matching import map_match_trips
from src.ingestion.pipeline import parse_ev_payload
from src.trips.carbon import trips_with_carbon_credits
from src.trips.segmentation import segment_trips


def main() -> None:
    root = Path(__file__).resolve().parent
    raw_csv = root / "data" / "raw" / "ev_prod_data.csv"
    out_dir = root / "outputs" / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_ev_payload(str(raw_csv))
    df, _masks = detect_gps_anomalies(df)
    df = smooth_latlon_kalman(df)
    generate_gps_anomaly_report(df, out_dir=out_dir)

    devs = top_devices_for_map(df, 2)
    if devs:
        write_raw_vs_smoothed_map(df, device_ids=devs, out_html=out_dir / "gps_raw_vs_smoothed_map.html")
        write_raw_vs_smoothed_map(df, device_ids=devs, out_html=out_dir / "gps_map.html")
    else:
        (out_dir / "gps_map.html").write_text("<html><body>No GPS points for map.</body></html>", encoding="utf-8")

    trips = segment_trips(df)
    credits = trips_with_carbon_credits(trips)

    trips.to_csv(out_dir / "trip_segments.csv", index=False)
    credits.to_csv(out_dir / "trip_carbon_credits.csv", index=False)

    if os.environ.get("ENABLE_MAP_MATCHING", "0") == "1":
        mm = map_match_trips(df, trips, use_smoothed=True)
        mm.to_csv(out_dir / "trip_route_matching.csv", index=False)
    else:
        print("Map-matching disabled (set ENABLE_MAP_MATCHING=1 to generate trip_route_matching.csv)")

    print("Task 3 completed:", out_dir)


if __name__ == "__main__":
    main()
