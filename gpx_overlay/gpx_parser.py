"""Utilities for reading and preparing GPX data."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
import gpxpy
import numpy as np
import pytz
from scipy.interpolate import UnivariateSpline


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two points in metres."""
    R = 6371000
    phi1, phi2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorised haversine (arrays in degrees)."""
    R = 6371000.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def parse_gpx(filepath: str):
    points = []
    gpx_start_time = None
    gpx_end_time = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)

        points_iter = (
            pt
            for track in gpx.tracks
            for segment in track.segments
            for pt in segment.points
        )
        for point in points_iter:
            if (
                point.time
                and point.latitude is not None
                and point.longitude is not None
                and point.elevation is not None
            ):
                hr_val = None
                for ext in getattr(point, "extensions", []):
                    hr_node = ext.find('.//{*}hr')
                    if hr_node is not None and hr_node.text:
                        hr_val = float(hr_node.text.strip())
                        break
                points.append(
                    {
                        "time": point.time,
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "ele": point.elevation,
                        "hr": hr_val,
                    }
                )
        if points:
            gpx_start_time = points[0]["time"]
            gpx_end_time = points[-1]["time"]
    except Exception as e:  # pragma: no cover - logging only
        print(f"Erreur GPX: {e}")
        return [], None, None
    return points, gpx_start_time, gpx_end_time


def filter_points_by_time(points, start_time_str, duration_seconds, timezone_str):
    tz = pytz.timezone(timezone_str)
    start_time = tz.localize(datetime.fromisoformat(start_time_str))
    end_time = start_time + timedelta(seconds=duration_seconds)
    filtered = [pt for pt in points if start_time <= pt["time"].astimezone(tz) <= end_time]
    if len(filtered) < 2:
        raise ValueError("Pas assez de points GPX pour la plage horaire spécifiée.")
    return filtered, start_time, tz


def interpolate_data(times, data, interp_times, s: float = 0):
    unique_times, unique_indices = np.unique(times, return_index=True)
    unique_data = data[unique_indices]
    if len(unique_times) < 2:
        return np.interp(interp_times, times, data)
    k_value = min(3, len(unique_times) - 1)
    if k_value < 1:
        return np.full_like(interp_times, data[0] if len(data) > 0 else 0)
    spline = UnivariateSpline(unique_times, unique_data, k=k_value, s=s)
    return spline(interp_times)


def pace_min_per_km_from_speed_kmh(speed_kmh: float) -> float:
    """Allure (min/km) depuis vitesse (km/h). Retourne np.inf si vitesse très faible."""
    if speed_kmh is None or speed_kmh <= 0.05:
        return float("inf")
    return 60.0 / float(speed_kmh)


def format_pace_mmss(pace_min_per_km: float) -> str:
    """Formate allure en m:ss/km. Inf -> '—'."""
    if not np.isfinite(pace_min_per_km) or pace_min_per_km > 59:
        return "—"
    m = int(pace_min_per_km)
    s = int(round((pace_min_per_km - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d} /km"


def prepare_track_arrays(times_seconds, lats, lons, eles, hrs_raw, clip_duration, fps):
    """Pré-calcul des séries interpolées communes aux rendus."""
    dists = haversine_np(lats[:-1], lons[:-1], lats[1:], lons[1:])
    dt = np.diff(times_seconds)
    segment_speeds = np.where(dt > 0, (dists / dt) * 3.6, 0.0)
    if segment_speeds.size:
        speeds = np.insert(segment_speeds, 0, segment_speeds[0])
    else:
        speeds = np.array([0.0])

    total_frames = max(1, int(clip_duration * fps))
    interp_times = np.linspace(0.0, float(clip_duration), num=total_frames, endpoint=False)
    interp_lats = interpolate_data(times_seconds, lats, interp_times)
    interp_lons = interpolate_data(times_seconds, lons, interp_times)
    interp_eles = interpolate_data(times_seconds, eles, interp_times)
    interp_speeds = np.clip(
        interpolate_data(times_seconds, speeds, interp_times), 0.0, None
    )

    if len(interp_eles) > 1:
        dist_interp = haversine_np(interp_lats[:-1], interp_lons[:-1], interp_lats[1:], interp_lons[1:])
        elev_diff = np.diff(interp_eles)
        seg_slopes = np.where(dist_interp > 0, elev_diff / dist_interp, 0.0) * 100.0
        slopes = np.insert(seg_slopes, 0, seg_slopes[0] if seg_slopes.size else 0.0)
        if slopes.size >= 7:
            kernel = np.ones(7) / 7.0
            interp_slopes = np.convolve(slopes, kernel, mode="same")
        else:
            interp_slopes = slopes
    else:
        interp_slopes = np.zeros_like(interp_eles)

    interp_pace = np.where(interp_speeds > 0.05, 60.0 / interp_speeds, np.inf)
    if np.isfinite(hrs_raw).sum() >= 2:
        mask = np.isfinite(hrs_raw)
        interp_hrs = interpolate_data(times_seconds[mask], hrs_raw[mask], interp_times)
    elif np.isfinite(hrs_raw).sum() == 1:
        val = float(hrs_raw[np.isfinite(hrs_raw)][0])
        interp_hrs = np.full_like(interp_times, val, dtype=float)
    else:
        interp_hrs = np.full_like(interp_times, np.nan, dtype=float)

    return {
        "total_frames": total_frames,
        "interp_times": interp_times,
        "interp_lats": interp_lats,
        "interp_lons": interp_lons,
        "interp_eles": interp_eles,
        "interp_speeds": interp_speeds,
        "interp_slopes": interp_slopes,
        "interp_pace": interp_pace,
        "interp_hrs": interp_hrs,
    }


def format_hms(seconds: int) -> str:
    """Format seconds into H:MM:SS."""
    return str(timedelta(seconds=int(max(0, seconds))))


__all__ = [
    "haversine",
    "haversine_np",
    "parse_gpx",
    "filter_points_by_time",
    "interpolate_data",
    "pace_min_per_km_from_speed_kmh",
    "format_pace_mmss",
    "prepare_track_arrays",
    "format_hms",
]
