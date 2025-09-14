"""Helpers for dealing with map tiles and WebMercator math."""
from __future__ import annotations

import math
import numpy as np

MERCATOR_LAT_MAX = 85.05112878


def _clamp_lat(lat: float) -> float:
    return max(-MERCATOR_LAT_MAX, min(MERCATOR_LAT_MAX, lat))


def lonlat_to_pixel(lon: float, lat: float, zoom: int):
    """Coordonnées pixels monde (x,y) au zoom donné (tuile 256px)."""
    lat = _clamp_lat(lat)
    n = 256 * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * n
    siny = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * n
    return x, y


def lonlat_to_pixel_np(lons, lats, zoom: int):
    """Vectorized lon/lat -> pixel conversion."""
    lats = np.clip(lats, -MERCATOR_LAT_MAX, MERCATOR_LAT_MAX)
    n = 256 * (2 ** zoom)
    xs = (lons + 180.0) / 360.0 * n
    siny = np.sin(np.radians(lats))
    ys = (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi)) * n
    return xs, ys


def bbox_fit_zoom(width_px: int, height_px: int, lon_min, lat_min, lon_max, lat_max, padding_px: int = 0) -> int:
    """Renvoie le zoom entier max qui FAIT TENIR la bbox dans width/height (padding appliqué)."""
    width_in = max(1, width_px - 2 * padding_px)
    height_in = max(1, height_px - 2 * padding_px)
    lo, hi = 1, 19
    best = 1
    while lo <= hi:
        z = (lo + hi) // 2
        x1, y1 = lonlat_to_pixel(lon_min, lat_min, z)
        x2, y2 = lonlat_to_pixel(lon_max, lat_max, z)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx <= width_in and dy <= height_in:
            best = z
            lo = z + 1
        else:
            hi = z - 1
    return best


def zoom_level_ui_to_offset(level_ui: int) -> int:
    """Map 1..12 -> décalage de zoom par rapport à l'ajustement bbox. 8=0, 12=+4, 1=-7."""
    level_ui = max(1, min(12, int(level_ui)))
    return level_ui - 8


def make_static_map(width: int, height: int, url_template: str):
    """Return a StaticMap instance with retry logic for CyclOSM tiles."""
    from staticmap import StaticMap  # type: ignore

    class RetryingStaticMap(StaticMap):
        def get(self, url, **kwargs):  # pragma: no cover - network
            try:
                status, content = super().get(url, **kwargs)
            except Exception:  # pragma: no cover
                status, content = 0, None
            if status == 404 and "tile-cyclosm.openstreetmap.fr" in url:
                for sub in "abc":
                    alt = url.replace("//a.tile-cyclosm", f"//{sub}.tile-cyclosm")
                    try:
                        status, content = super().get(alt, **kwargs)
                    except Exception:  # pragma: no cover
                        status, content = 0, None
                    if status == 200:
                        break
                if status in (404, 0):
                    alt = url.replace(
                        "a.tile-cyclosm.openstreetmap.fr/cyclosm",
                        "tile.openstreetmap.org",
                    )
                    try:
                        status, content = super().get(alt, **kwargs)
                    except Exception:  # pragma: no cover
                        status, content = 0, None
            return status, content

    return RetryingStaticMap(width, height, url_template=url_template, padding_x=0, padding_y=0, delay_between_retries=1)


__all__ = [
    "MERCATOR_LAT_MAX",
    "_clamp_lat",
    "lonlat_to_pixel",
    "lonlat_to_pixel_np",
    "bbox_fit_zoom",
    "zoom_level_ui_to_offset",
    "make_static_map",
]
