from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from urllib.error import HTTPError, URLError


@dataclass(frozen=True)
class MatchResult:
    matched_route_km: float
    confidence: float
    status: str


def _downsample_points(points: list[tuple[float, float]], max_points: int = 100) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, num=max_points).round().astype(int)
    return [points[i] for i in idx]


def _dedupe_consecutive(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return points
    out = [points[0]]
    for p in points[1:]:
        if p != out[-1]:
            out.append(p)
    return out


def _osrm_match_distance_km(
    latlon_points: list[tuple[float, float]],
    *,
    base_url: str = "https://router.project-osrm.org",
    profile: str = "driving",
    max_points: int = 100,
    timeout_sec: int = 20,
) -> MatchResult:
    """
    Uses OSRM public match service:
      GET /match/v1/{profile}/{coords}?geometries=geojson&overview=false&steps=false
    Returns matched distance in km.
    """
    
    max_points = int(min(max_points, 100))
    pts = _downsample_points(latlon_points, max_points=max_points)
    if len(pts) < 2:
        return MatchResult(matched_route_km=float("nan"), confidence=0.0, status="not_enough_points")

    coords = ";".join([f"{lon:.6f},{lat:.6f}" for lat, lon in pts])
    query = urllib.parse.urlencode(
        {
            "overview": "false",
            "geometries": "geojson",
            "steps": "false",
            "tidy": "true",
        }
    )
    url = f"{base_url.rstrip('/')}/match/v1/{profile}/{coords}?{query}"

    req = urllib.request.Request(url, headers={"User-Agent": "renewcred-task3/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        body_snip = body[:120].replace("\n", " ")
        return MatchResult(
            matched_route_km=float("nan"),
            confidence=0.0,
            status=f"http_{int(getattr(e, 'code', 0) or 0)}:{body_snip}",
        )
    except URLError as e:
        return MatchResult(matched_route_km=float("nan"), confidence=0.0, status=f"url_error:{type(e).__name__}")
    except Exception as e:
        return MatchResult(matched_route_km=float("nan"), confidence=0.0, status=f"request_error:{type(e).__name__}")

    if not isinstance(payload, dict) or payload.get("code") != "Ok":
        code = payload.get("code") if isinstance(payload, dict) else "invalid_json"
        return MatchResult(matched_route_km=float("nan"), confidence=0.0, status=f"osrm_{code}")

    matchings = payload.get("matchings") or []
    if not matchings:
        return MatchResult(matched_route_km=float("nan"), confidence=0.0, status="no_matchings")

    m0 = matchings[0]
    dist_m = float(m0.get("distance", float("nan")))
    conf = float(m0.get("confidence", 0.0)) if "confidence" in m0 else 0.0
    if not np.isfinite(dist_m):
        return MatchResult(matched_route_km=float("nan"), confidence=conf, status="missing_distance")
    return MatchResult(matched_route_km=dist_m / 1000.0, confidence=conf, status="ok")


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return float(r * c)


def raw_track_km(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    dist = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points[:-1], points[1:]):
        dist += _haversine_km(lat1, lon1, lat2, lon2)
    return float(dist)


def map_match_trips(
    df_points: pd.DataFrame,
    trips: pd.DataFrame,
    *,
    use_smoothed: bool = True,
    # OSRM public match commonly rejects large traces; keep conservative.
    max_points_per_trip: int = 50,
    rate_limit_sec: float = 0.35,
) -> pd.DataFrame:
    """
    Given point-level telemetry and a trip summary table, calls OSRM match per trip and returns
    per-trip discrepancy metrics.

    Points are selected by (device_id, ts in [start_ts, end_ts]) for each trip.
    """
    if trips.empty:
        return pd.DataFrame(
            columns=[
                "device_id",
                "trip_id",
                "raw_gps_km",
                "matched_route_km",
                "discrepancy_km",
                "discrepancy_pct",
                "match_confidence",
                "match_status",
            ]
        )

    d = df_points.copy()
    d["ts"] = pd.to_datetime(d.get("ts"), utc=True, errors="coerce")
    d = d.dropna(subset=["device_id", "ts"]).sort_values(["device_id", "ts"], kind="mergesort")

    lat_col = "gps_lat_smooth" if use_smoothed and "gps_lat_smooth" in d.columns else "gps_lat"
    lon_col = "gps_lon_smooth" if use_smoothed and "gps_lon_smooth" in d.columns else "gps_lon"
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")

    rows: list[dict[str, Any]] = []
    for _, tr in trips.iterrows():
        dev = tr.get("device_id")
        tid = tr.get("trip_id")
        start_ts = pd.to_datetime(tr.get("start_ts"), utc=True, errors="coerce")
        end_ts = pd.to_datetime(tr.get("end_ts"), utc=True, errors="coerce")
        seg = d[(d["device_id"] == dev) & (d["ts"] >= start_ts) & (d["ts"] <= end_ts)]
        pts = [
            (float(r[lat_col]), float(r[lon_col]))
            for _, r in seg.iterrows()
            if pd.notna(r[lat_col]) and pd.notna(r[lon_col])
        ]
        pts = _dedupe_consecutive(pts)
        pts = _downsample_points(pts, max_points=max_points_per_trip)

        raw_km = raw_track_km(pts)

        match: MatchResult | None = None
        for cap in [max_points_per_trip, 40, 30, 25, 20, 15]:
            cap = int(min(cap, max_points_per_trip))
            match = _osrm_match_distance_km(pts, max_points=cap)
            if match.status == "ok":
                break
            if match.status.startswith("http_414") or match.status.startswith("http_400"):
                continue
            if match.status.startswith("http_429"):
                time.sleep(max(1.0, rate_limit_sec * 4))
                continue
            break
        assert match is not None

        matched_km = match.matched_route_km
        disc = float(matched_km - raw_km) if np.isfinite(matched_km) else float("nan")
        disc_pct = float(disc / raw_km * 100.0) if np.isfinite(disc) and raw_km > 1e-6 else float("nan")

        rows.append(
            {
                "device_id": dev,
                "trip_id": tid,
                "raw_gps_km": round(raw_km, 4),
                "matched_route_km": round(matched_km, 4) if np.isfinite(matched_km) else np.nan,
                "discrepancy_km": round(disc, 4) if np.isfinite(disc) else np.nan,
                "discrepancy_pct": round(disc_pct, 2) if np.isfinite(disc_pct) else np.nan,
                "match_confidence": round(match.confidence, 4),
                "match_status": match.status,
            }
        )

        time.sleep(rate_limit_sec)

    return pd.DataFrame(rows)

