from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import List, Tuple

import gpxpy
import numpy as np
import pytz
from scipy.interpolate import UnivariateSpline
import xml.etree.ElementTree as ET


@dataclass
class TrackData:
    """Contient les données interpolées nécessaires au rendu."""

    start_time: datetime
    timezone: tzinfo
    times_seconds: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    eles: np.ndarray
    hrs_raw: np.ndarray
    interp_times: np.ndarray
    interp_lats: np.ndarray
    interp_lons: np.ndarray
    interp_eles: np.ndarray
    interp_speeds: np.ndarray
    interp_slopes: np.ndarray
    interp_pace: np.ndarray
    interp_hrs: np.ndarray
    total_frames: int
    lat_min_raw: float
    lat_max_raw: float
    lon_min_raw: float
    lon_max_raw: float


def parse_gpx(filepath: str) -> Tuple[List[dict], datetime | None, datetime | None]:
    points: List[dict] = []
    gpx_start_time: datetime | None = None
    gpx_end_time: datetime | None = None
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
    except Exception as e:
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


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized haversine (arrays in degrees)."""
    R = 6371000.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_interpolated_track(
    gpx_filename: str,
    start_offset: int,
    clip_duration: int,
    fps: int,
    timezone: str = "Europe/Paris",
) -> TrackData:
    points, gpx_start, _ = parse_gpx(gpx_filename)
    if not points:
        raise ValueError("Aucun point GPX.")
    if gpx_start is None:
        raise ValueError("Horodatage GPX invalide")

    tz = pytz.timezone(timezone)
    start_time = gpx_start + timedelta(seconds=start_offset)
    start_str = start_time.astimezone(tz).replace(tzinfo=None).isoformat()
    filtered_points, start_time_localized, tzinfo = filter_points_by_time(points, start_str, clip_duration, timezone)

    times_seconds = np.array([(pt["time"] - start_time_localized).total_seconds() for pt in filtered_points], dtype=float)
    lats = np.array([pt["lat"] for pt in filtered_points], dtype=float)
    lons = np.array([pt["lon"] for pt in filtered_points], dtype=float)
    eles = np.array([pt["ele"] for pt in filtered_points], dtype=float)
    hrs_raw = np.array([
        (pt.get("hr") if pt.get("hr") is not None else np.nan) for pt in filtered_points
    ], dtype=float)

    data = prepare_track_arrays(times_seconds, lats, lons, eles, hrs_raw, clip_duration, fps)

    return TrackData(
        start_time=start_time_localized,
        timezone=tzinfo,
        times_seconds=times_seconds,
        lats=lats,
        lons=lons,
        eles=eles,
        hrs_raw=hrs_raw,
        interp_times=data["interp_times"],
        interp_lats=data["interp_lats"],
        interp_lons=data["interp_lons"],
        interp_eles=data["interp_eles"],
        interp_speeds=data["interp_speeds"],
        interp_slopes=data["interp_slopes"],
        interp_pace=data["interp_pace"],
        interp_hrs=data["interp_hrs"],
        total_frames=data["total_frames"],
        lat_min_raw=float(np.min(lats)),
        lat_max_raw=float(np.max(lats)),
        lon_min_raw=float(np.min(lons)),
        lon_max_raw=float(np.max(lons)),
    )


__all__ = ["TrackData", "parse_gpx", "load_interpolated_track"]
