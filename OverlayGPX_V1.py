# Overlay_dynamique.py
# GPX -> Vidéo avec carte fond (StaticMap), zoom 1..12 avec centre/zoom explicites,
# emprise large fiable (WebMercator), durée par défaut 20s.

import json
import math
import os
from datetime import datetime, timedelta
import sys, time
import threading
import imageio.v2 as imageio

import gpxpy
import numpy as np
import pytz
from PIL import Image, ImageDraw, ImageFont, ImageTk
from scipy.interpolate import UnivariateSpline
import xml.etree.ElementTree as ET  # <-- ajouté pour lire la FC dans les extensions GPX

class TileFetchError(Exception):
    """Erreur bloquante lors du téléchargement de tuiles."""
    pass


# --- Tuiles/cartes (optionnel) ---
try:
    from staticmap import StaticMap  # type: ignore
    STATICMAP_AVAILABLE = True
except Exception:
    STATICMAP_AVAILABLE = False

# --- Styles de cartes disponibles ---
MAP_TILE_SERVERS = {
    "Aucun": None,
    "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "Satellite ESRI": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "CyclOSM (FR)": "https://a.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png",
    "CyclOSM Forest": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
    "OpenSnowMap": "https://tiles.opensnowmap.org/pistes/{z}/{x}/{y}.png",
    
}

# Zoom maximal supporté par chaque fournisseur de tuiles
MAX_ZOOM = {
    "OpenStreetMap": 19,
    "Satellite ESRI": 19,
    "CyclOSM (FR)": 19,
    "CyclOSM Forest": 17,
    "OpenSnowMap": 18,
}

# --- Réglages géométrie et apparence ---
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_FPS = 25
DEFAULT_FONT_PATH = "arial.ttf"
DEFAULT_CLIP_DURATION_SECONDS = 5

# Couleurs par défaut
BG_COLOR = (0, 0, 0)
PATH_COLOR = (196, 152, 29)
CURRENT_PATH_COLOR = (240, 179, 10)
CURRENT_POINT_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
GAUGE_BG_COLOR = (30, 30, 30)

FONT_SIZE_LARGE = 40
FONT_SIZE_MEDIUM = 20
GRAPH_FONT_SCALE = 0.7
MIN_GRAPH_FONT_SIZE = 4
MARGIN = 10
GRAPH_PADDING = 100

# Intervalle fixe pour le graphe d'allure (min/km)
PACE_GRAPH_MIN = 2.0
PACE_GRAPH_MAX = 8.0
# Durée de lissage (secondes) par défaut appliquée aux données des graphes
DEFAULT_GRAPH_SMOOTHING_SECONDS = 20.0

LEFT_COLUMN_WIDTH = 480
RIGHT_COLUMN_X = 1500
RIGHT_COLUMN_WIDTH = 400
MAP_HEIGHT_DEFAULT = 480
MAP_BOTTOM = MARGIN + MAP_HEIGHT_DEFAULT
COMPASS_HEIGHT = 70
COMPASS_Y = 1000
INFO_LINES_COUNT = 6
INFO_LINE_SPACING = FONT_SIZE_LARGE + 10
INFO_TEXT_HEIGHT = INFO_LINES_COUNT * INFO_LINE_SPACING
INFO_TEXT_Y = 450
GAUGE_BASE_Y = 900


def compute_graph_font_size(medium_size: int) -> int:
    """Calcule une taille de police dédiée aux graphiques, réduite d'environ 30 %."""

    scaled_size = int(medium_size * GRAPH_FONT_SCALE)
    return max(MIN_GRAPH_FONT_SIZE, scaled_size)

DEFAULT_ELEMENT_CONFIGS = {
    "Carte": {
        "visible": True,
        "x": MARGIN,
        "y": MARGIN,
        "width": 300,
        "height": 300,
    },
    "Profil Altitude": {
        "visible": True,
        "x": RIGHT_COLUMN_X,
        "y": 400,
        "width": RIGHT_COLUMN_WIDTH,
        "height": 130,
    },
    "Profil Vitesse": {
        "visible": True,
        "x": RIGHT_COLUMN_X,
        "y": 600,
        "width": RIGHT_COLUMN_WIDTH,
        "height": 100,
    },
    # --- AJOUTS : profils Allure & Cardio ---
    "Profil Allure": {
        "visible": True,
        "x": RIGHT_COLUMN_X,
        "y": 750,
        "width": RIGHT_COLUMN_WIDTH,
        "height": 100,
    },
    "Profil Cardio": {
        "visible": True,
        "x": RIGHT_COLUMN_X,
        "y": 920,
        "width": RIGHT_COLUMN_WIDTH,
        "height": 100,
    },
    "Jauge Vitesse Circulaire": {
        "visible": False,
        "x": 60,
        "y": 1010,
        "width": 300,
        "height": 50,
    },
    "Jauge Vitesse Linéaire": {
        "visible": True,
        "x": 705,
        "y": 936,
        "width": 500,
        "height": 30,
    },
    "Compteur de vitesse": {
        "visible": True,
        "x": MARGIN,
        "y": GAUGE_BASE_Y,
        "width": 400,
        "height": 160,
    },
    "Boussole (ruban)": {
        "visible": True,
        "x": 704,
        "y": -100,
        "width": 513,  # largeur personnalisée du ruban
        "height": 70,
    },
    # Augmente la hauteur pour accueillir Allure, FC et Pente
    "Infos Texte": {
        "visible": True,
        "x": MARGIN,
        "y": INFO_TEXT_Y,
        "width": 368,
        "height": INFO_TEXT_HEIGHT,  # 6 lignes : Vitesse, Altitude, Heure, Pente, Allure, FC
    },
}

# ---------- Utils GPX & maths ----------

def rgb_to_hex(rgb_tuple):
    if isinstance(rgb_tuple, str) and rgb_tuple.startswith("#"):
        return rgb_tuple
    try:
        r, g, b = rgb_tuple
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    except Exception:
        return "#000000"

def hex_to_rgb(value):
    if isinstance(value, tuple):
        return value
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def darken_color(color, factor=0.7):
    r, g, b = hex_to_rgb(color)
    return (int(r * factor), int(g * factor), int(b * factor))

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    phi1, phi2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized haversine (arrays in degrees)."""
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
    except Exception as e:
        print(f"Erreur GPX: {e}")
        return [], None, None
    return points, gpx_start_time, gpx_end_time

def filter_points_by_time(
    points,
    start_time_str,
    duration_seconds,
    timezone_str,
    pre_margin_seconds: float = 0.0,
    post_margin_seconds: float = 0.0,
):
    tz = pytz.timezone(timezone_str)
    clip_start = tz.localize(datetime.fromisoformat(start_time_str))

    if not points:
        raise ValueError("Aucun point GPX disponible.")

    tz_times = [pt["time"].astimezone(tz) for pt in points]
    earliest_time = min(tz_times)
    latest_time = max(tz_times)

    clip_end = clip_start + timedelta(seconds=duration_seconds)
    if clip_end > latest_time:
        raise ValueError("La durée du clip dépasse la fin du GPX.")

    pre_margin = max(0.0, float(pre_margin_seconds))
    post_margin = max(0.0, float(post_margin_seconds))

    extended_start = clip_start - timedelta(seconds=pre_margin)
    if extended_start < earliest_time:
        extended_start = earliest_time

    extended_end = clip_end + timedelta(seconds=post_margin)
    if extended_end > latest_time:
        extended_end = latest_time

    if extended_end <= extended_start:
        raise ValueError("Plage temporelle invalide pour le GPX fourni.")

    filtered = [
        pt
        for pt in points
        if extended_start <= pt["time"].astimezone(tz) <= extended_end
    ]
    if len(filtered) < 2:
        raise ValueError("Pas assez de points GPX pour la plage horaire spécifiée.")

    total_duration = (extended_end - extended_start).total_seconds()
    clip_start_offset = max(0.0, (clip_start - extended_start).total_seconds())
    clip_end_offset = min(total_duration, (clip_end - extended_start).total_seconds())

    return (
        filtered,
        extended_start,
        tz,
        clip_start_offset,
        clip_end_offset,
        total_duration,
    )

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

# ---------- Lissage ----------

def smooth_series(values: np.ndarray, window_size: int) -> np.ndarray:
    """Lisse un signal 1D via moyenne glissante en ignorant les valeurs non finies."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window_size <= 1:
        return arr.copy()

    if window_size > arr.size:
        window_size = arr.size
    if window_size > 1 and window_size % 2 == 0:
        window_size -= 1
    if window_size <= 1:
        return arr.copy()

    kernel = np.ones(window_size, dtype=float)
    finite_mask = np.isfinite(arr)
    weighted_sum = np.convolve(np.where(finite_mask, arr, 0.0), kernel, mode="same")
    weight_counts = np.convolve(finite_mask.astype(float), kernel, mode="same")

    smoothed = np.divide(
        weighted_sum,
        weight_counts,
        out=np.full(arr.shape, np.nan, dtype=float),
        where=weight_counts > 0,
    )
    return np.where(weight_counts > 0, smoothed, arr)

# ---------- Helpers Allure/FC ----------

def pace_min_per_km_from_speed_kmh(speed_kmh: float) -> float:
    """Allure (min/km) depuis vitesse (km/h). Retourne np.inf si vitesse très faible."""
    if speed_kmh is None or speed_kmh <= 0.05:
        return float('inf')
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


def prepare_track_arrays(
    times_seconds,
    lats,
    lons,
    eles,
    hrs_raw,
    total_duration_seconds,
    fps,
    smoothing_seconds: float = DEFAULT_GRAPH_SMOOTHING_SECONDS,
):
    """Pré-calcul des séries interpolées communes aux rendus avec lissage optionnel."""

    dists = haversine_np(lats[:-1], lons[:-1], lats[1:], lons[1:])
    dt = np.diff(times_seconds)
    segment_speeds = np.where(dt > 0, (dists / dt) * 3.6, 0.0)
    if segment_speeds.size:
        speeds = np.insert(segment_speeds, 0, segment_speeds[0])
    else:
        speeds = np.array([0.0])

    total_frames = max(1, int(total_duration_seconds * fps))
    interp_times = np.linspace(0.0, float(total_duration_seconds), num=total_frames, endpoint=False)
    interp_lats = interpolate_data(times_seconds, lats, interp_times)
    interp_lons = interpolate_data(times_seconds, lons, interp_times)
    interp_eles = interpolate_data(times_seconds, eles, interp_times)
    interp_speeds = np.clip(
        interpolate_data(times_seconds, speeds, interp_times), 0.0, None
    )

    smoothing_seconds = max(0.0, float(smoothing_seconds))
    smoothing_frames = 0
    if fps > 0 and smoothing_seconds > 0.0:
        smoothing_frames = int(round(smoothing_seconds * float(fps)))
        if smoothing_frames < 2:
            smoothing_frames = 0

    if smoothing_frames:
        interp_eles = smooth_series(interp_eles, smoothing_frames)
        interp_speeds = smooth_series(interp_speeds, smoothing_frames)

    if len(interp_eles) > 1:
        dist_interp = haversine_np(
            interp_lats[:-1], interp_lons[:-1], interp_lats[1:], interp_lons[1:]
        )
        elev_diff = np.diff(interp_eles)
        seg_slopes = np.where(dist_interp > 0, elev_diff / dist_interp, 0.0) * 100.0
        interp_slopes = np.insert(seg_slopes, 0, seg_slopes[0] if seg_slopes.size else 0.0)
    else:
        interp_slopes = np.zeros_like(interp_eles)

    if smoothing_frames and interp_slopes.size:
        interp_slopes = smooth_series(interp_slopes, smoothing_frames)

    interp_pace = np.where(interp_speeds > 0.05, 60.0 / interp_speeds, np.inf)
    if smoothing_frames and interp_pace.size:
        interp_pace = smooth_series(interp_pace, smoothing_frames)

    finite_hr_count = int(np.isfinite(hrs_raw).sum())
    if finite_hr_count >= 2:
        mask = np.isfinite(hrs_raw)
        interp_hrs = interpolate_data(times_seconds[mask], hrs_raw[mask], interp_times)
    elif finite_hr_count == 1:
        val = float(hrs_raw[np.isfinite(hrs_raw)][0])
        interp_hrs = np.full_like(interp_times, val, dtype=float)
    else:
        interp_hrs = np.full_like(interp_times, np.nan, dtype=float)

    if smoothing_frames and np.isfinite(interp_hrs).sum() >= 1:
        interp_hrs = smooth_series(interp_hrs, smoothing_frames)

    return {
        "total_frames": total_frames,
        "duration_seconds": float(total_duration_seconds),
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

def _norm_heading_deg(a: float) -> float:
    """Normalise un angle en degrés dans [0, 360)."""
    a = float(a)
    a = a % 360.0
    return a if a >= 0 else a + 360.0


# ---------- WebMercator helpers (clé pour une emprise fiable) ----------

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
    # Recherche par dichotomie sur zoom [1..19]
    lo, hi = 1, 19
    best = 1
    while lo <= hi:
        z = (lo + hi) // 2
        x1, y1 = lonlat_to_pixel(lon_min, lat_min, z)
        x2, y2 = lonlat_to_pixel(lon_max, lat_max, z)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx <= width_in and dy <= height_in:
            best = z  # on peut encore zoomer + (plus proche)
            lo = z + 1
        else:
            hi = z - 1
    return best

def zoom_level_ui_to_offset(level_ui: int) -> int:
    """Map 1..12 -> décalage de zoom par rapport à l'ajustement bbox. 8=0, 12=+4, 1=-7."""
    level_ui = max(1, min(12, int(level_ui)))
    return level_ui - 8

def make_static_map(width: int, height: int, url_template: str, max_zoom: int = 19):
    """Return a StaticMap instance with retry logic and zoom clamping."""
    from staticmap import StaticMap  # type: ignore

    class RetryingStaticMap(StaticMap):
        def __init__(self, *args, **kwargs):
            self.max_zoom = kwargs.pop("max_zoom", 19)
            super().__init__(*args, **kwargs)

        def get(self, url, **kwargs):
            # Récupération de tuile avec fallback CyclOSM -> OSM
            try:
                status, content = super().get(url, **kwargs)
            except Exception:
                status, content = 0, None

            if status != 200 and "tile-cyclosm.openstreetmap.fr" in url:
                for sub in "abc":
                    alt = url.replace("//a.tile-cyclosm", f"//{sub}.tile-cyclosm")
                    try:
                        status, content = super().get(alt, **kwargs)
                    except Exception:
                        status, content = 0, None
                    if status == 200:
                        break
                if status != 200:
                    alt = url.replace(
                        "a.tile-cyclosm.openstreetmap.fr/cyclosm",
                        "tile.openstreetmap.org",
                    )
                    try:
                        status, content = super().get(alt, **kwargs)
                    except Exception:
                        status, content = 0, None

            # ⛔️ FAIL-FAST : si statut <> 200, on lève tout de suite
            if status != 200:
                raise TileFetchError(f"Échec téléchargement tuile: {url}")

            return status, content

        def render(self, zoom, center=None):
            # On borne le zoom et on ne masque plus l’erreur
            zoom = min(zoom, self.max_zoom)
            try:
                return super().render(zoom=zoom, center=center)
            except TileFetchError:
                # Déjà une erreur explicite de tuile -> on propage
                raise
            except Exception as e:
                # On convertit toute autre erreur en TileFetchError
                raise TileFetchError(str(e)) from e

    return RetryingStaticMap(
        width,
        height,
        url_template=url_template,
        padding_x=0,
        padding_y=0,
        delay_between_retries=1,
        max_zoom=max_zoom,
    )


def render_base_map(width: int, height: int, map_style: str, zoom: int,
                    lon_c: float, lat_c: float, bg_c: tuple[int, int, int],
                    fail_on_tile_error: bool = True):
    """Render une carte de fond. Si fail_on_tile_error=True, lève TileFetchError si une tuile échoue."""
    if map_style == "Aucun" or not STATICMAP_AVAILABLE:
        from PIL import Image
        return Image.new("RGB", (width, height), bg_c)

    template = MAP_TILE_SERVERS.get(map_style) or MAP_TILE_SERVERS["OpenStreetMap"]
    max_zoom = MAX_ZOOM.get(map_style, MAX_ZOOM["OpenStreetMap"])
    center = (lon_c, _clamp_lat(lat_c))

    def _render_one(url_template):
        s = make_static_map(width, height, url_template, max_zoom=max_zoom)
        return s.render(zoom=zoom, center=center).convert("RGB")

    try:
        if isinstance(template, (list, tuple)):
            # couche de base
            base_img = _render_one(template[0]).convert("RGBA")
            # couches additionnelles
            for overlay_url in template[1:]:
                overlay_img = _render_one(overlay_url).convert("RGBA")
                base_img.paste(overlay_img, (0, 0), overlay_img)
            return base_img.convert("RGB")
        else:
            return _render_one(template)
    except Exception as e:
        if fail_on_tile_error:
            if not isinstance(e, TileFetchError):
                e = TileFetchError(str(e))
            raise e
        # Fallback facultatif (non utilisé si fail_on_tile_error=True)
        from PIL import Image
        return Image.new("RGB", (width, height), bg_c)

# ---------- Graph helpers ----------

def auto_speed_bounds(speeds: np.ndarray) -> tuple[float, float]:
    """Determine appropriate speed axis bounds based on observed speeds."""
    max_speed = float(np.max(speeds)) if speeds.size else 0.0
    if max_speed < 25:
        return 0.0, 25.0  # course à pied
    if max_speed < 80:
        return 0.0, 80.0  # vélo
    return 0.0, float(math.ceil(max_speed / 20.0) * 20.0)  # ski ou autres sports rapides

class GraphTransformer:
    def __init__(self, data_min: float, data_max: float, draw_area: dict):
        self.data_min, self.data_max = float(data_min), float(data_max)
        self.draw_x, self.draw_y, self.draw_width, self.draw_height = (
            draw_area["x"], draw_area["y"], draw_area["width"], draw_area["height"]
        )

    def to_xy(self, index: int, value: float, total_points: int):
        data_range = self.data_max - self.data_min
        x = self.draw_x + int(index / (total_points - 1) * self.draw_width) if total_points > 1 else self.draw_x
        y = self.draw_y + int((self.data_max - value) / (data_range + 1e-10) * self.draw_height)
        return (x, y)

def draw_graph(
    draw, path_coords, current_index, min_val, max_val, draw_area, font, title, unit,
    base_color, current_point_color, text_color, point_size: int = 4,
):
    # Graduations en ordonnée sous la courbe
    nb_ticks = 4
    for i in range(nb_ticks + 1):
        val = min_val + (max_val - min_val) * i / nb_ticks
        y = draw_area["y"] + int((max_val - val) / ((max_val - min_val) + 1e-10) * draw_area["height"])
        draw.line([(draw_area["x"], y), (draw_area["x"] + draw_area["width"], y)], fill=(80, 80, 80), width=1)
        val_str = f"{val:.0f}"
        text_bbox = draw.textbbox((0, 0), val_str, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.text((draw_area["x"] - text_w - 10, y - text_h / 2), val_str, font=font, fill=text_color)
    future_color = darken_color(base_color, 0.8)
    draw.line(path_coords, fill=future_color, width=4)
    if current_index < len(path_coords):
        draw.line(path_coords[: current_index + 1], fill=base_color, width=5)
        current_xy = path_coords[current_index]
        draw.ellipse(
            (current_xy[0]-point_size, current_xy[1]-point_size, current_xy[0]+point_size, current_xy[1]+point_size),
            fill=current_point_color,
        )

    # Légendes min/max (entiers pour rester simple)
    draw.text((draw_area["x"], draw_area["y"] + draw_area["height"] + 10), f"Min {title}: {min_val:.0f} {unit}", font=font, fill=text_color)
    draw.text((draw_area["x"] + draw_area["width"] - 200, draw_area["y"] + draw_area["height"] + 10), f"Max {title}: {max_val:.0f} {unit}", font=font, fill=text_color)


def is_area_visible(area: dict | None) -> bool:
    """Retourne True si la zone est visible et possède des dimensions non nulles."""
    if not area:
        return False
    return bool(area.get("visible", False) and area.get("width", 0) > 0 and area.get("height", 0) > 0)


def create_graph_background_image(
    resolution: tuple[int, int],
    path_coords: list[tuple[int, int]],
    min_val: float,
    max_val: float,
    draw_area: dict,
    font,
    title: str,
    unit: str,
    base_color,
    text_color,
) -> Image.Image:
    """Crée une image transparente contenant axes, graduations et courbe complète."""

    bg_img = Image.new("RGBA", resolution, (0, 0, 0, 0))
    if not is_area_visible(draw_area):
        return bg_img

    draw = ImageDraw.Draw(bg_img)

    x0 = int(draw_area.get("x", 0))
    y0 = int(draw_area.get("y", 0))
    width = int(draw_area.get("width", 0))
    height = int(draw_area.get("height", 0))

    nb_ticks = 4
    for i in range(nb_ticks + 1):
        val = min_val + (max_val - min_val) * i / nb_ticks
        y = y0 + int((max_val - val) / ((max_val - min_val) + 1e-10) * height)
        draw.line([(x0, y), (x0 + width, y)], fill=(80, 80, 80), width=1)
        val_str = f"{val:.0f}"
        text_bbox = draw.textbbox((0, 0), val_str, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.text((x0 - text_w - 10, y - text_h / 2), val_str, font=font, fill=text_color)

    future_color = darken_color(base_color, 0.8)
    if len(path_coords) >= 2:
        draw.line(path_coords, fill=future_color, width=4)
    elif len(path_coords) == 1:
        cx, cy = path_coords[0]
        draw.ellipse((cx - 1, cy - 1, cx + 1, cy + 1), fill=future_color)

    text_y = y0 + height + 10
    draw.text((x0, text_y), f"Min {title}: {min_val:.0f} {unit}", font=font, fill=text_color)
    draw.text((x0 + width - 200, text_y), f"Max {title}: {max_val:.0f} {unit}", font=font, fill=text_color)

    return bg_img


def draw_graph_progress_overlay(
    draw: ImageDraw.ImageDraw,
    path_coords: list[tuple[int, int]],
    current_index: int,
    base_color,
    current_point_color,
    point_size: int = 4,
) -> None:
    """Dessine la portion courante et le point actuel sur un graphe pré-dessiné."""

    if not path_coords:
        return

    idx = max(0, min(current_index, len(path_coords) - 1))
    if idx >= 1:
        draw.line(path_coords[: idx + 1], fill=base_color, width=5)

    cx, cy = path_coords[idx]
    draw.ellipse((cx - point_size, cy - point_size, cx + point_size, cy + point_size), fill=current_point_color)


def prepare_graph_layers(
    resolution: tuple[int, int],
    font,
    text_color,
    point_color,
    graph_specs: list[tuple[dict, list[tuple[int, int]], float, float, str, str, tuple[int, int, int]]],
) -> list[dict]:
    """Prépare les couches de graphes (fond pré-rendu + méta-données)."""

    layers: list[dict] = []
    for area, path_coords, min_val, max_val, title, unit, base_color in graph_specs:
        if not is_area_visible(area):
            continue
        background = create_graph_background_image(
            resolution,
            path_coords,
            min_val,
            max_val,
            area,
            font,
            title,
            unit,
            base_color,
            text_color,
        )
        layers.append(
            {
                "area": area,
                "path": path_coords,
                "min": min_val,
                "max": max_val,
                "title": title,
                "unit": unit,
                "base_color": base_color,
                "point_color": point_color,
                "background": background,
                "point_size": 4,
            }
        )
    return layers

def draw_circular_speedometer(draw, speed, speed_min, speed_max, draw_area, font, gauge_bg_color, text_color):
    x0, y0 = draw_area["x"], draw_area["y"]
    w, h = draw_area["width"], draw_area["height"]
    radius = min(w / 2.0, h) - 5
    cx = x0 + w / 2.0
    cy = y0 + h - 10
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.arc(bbox, start=180, end=360, fill=gauge_bg_color, width=12)
    fraction = 0.0 if speed_max <= speed_min else (speed - speed_min) / (speed_max - speed_min)
    fraction = max(0.0, min(1.0, fraction))
    col = (0, 255, 0) if fraction < 0.33 else ((255, 255, 0) if fraction < 0.66 else (255, 0, 0))
    draw.arc(bbox, start=180.0, end=180.0 + fraction * 180.0, fill=col, width=12)
    speed_text = f"{speed:.0f}"
    text_bbox = draw.textbbox((0, 0), speed_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]; text_h = text_bbox[3] - text_bbox[1]
    draw.text((cx - text_w/2, y0 + (h - text_h)/2 - 10), speed_text, font=font, fill=text_color)

def draw_linear_speedometer(draw, speed, speed_min, speed_max, draw_area, font, gauge_bg_color, text_color):
    x, y = draw_area["x"], draw_area["y"]
    w, h = draw_area["width"], draw_area["height"]
    draw.rectangle([x, y, x + w, y + h], outline=gauge_bg_color, width=2)
    fraction = 0.0 if speed_max <= speed_min else (speed - speed_min) / (speed_max - speed_min)
    fraction = max(0.0, min(1.0, fraction))
    col = (0, 255, 0) if fraction < 0.33 else ((255, 255, 0) if fraction < 0.66 else (255, 0, 0))
    fill_w = int(w * fraction)
    draw.rectangle([x, y, x + fill_w, y + h], fill=col)
    speed_text = f"{speed:.0f} km/h"
    text_bbox = draw.textbbox((0, 0), speed_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]; text_h = text_bbox[3] - text_bbox[1]
    draw.text((x + w/2 - text_w/2, y - h*2 - text_h/2), speed_text, font=font, fill=text_color)

def draw_digital_speedometer(draw, speed, speed_min, speed_max, draw_area, font, gauge_bg_color, text_color):
    """
    Compteur demi-cercle sans fond, avec décimation adaptative.
    - Ticks mineurs/majeurs
    - Libellés majeurs horizontaux (droits)
    - Aiguille rouge + pivot blanc
    - Les deux dernières valeurs majeures en rouge
    """
    import math

    # --- Réglage: laisser les libellés droits (horizontaux) ---
    ROTATE_LABELS = False

    x0, y0 = draw_area["x"], draw_area["y"]
    w, h = draw_area["width"], draw_area["height"]

    pad = 6
    radius = max(1.0, min(w / 2.0, h) - pad)
    cx = x0 + w / 2.0
    cy = y0 + h - pad

    if speed_max <= speed_min:
        return
    speed_clamped = max(speed_min, min(speed_max, speed))
    span = float(speed_max - speed_min)

    # Pas initiaux
    candidate_maj = [5, 10, 20, 25, 50]
    major_step = min(candidate_maj, key=lambda s: abs((span / s) - round(span / s)))
    if span / major_step > 16:
        major_step *= 2
    if span / major_step < 6:
        major_step = max(5, major_step // 2) if major_step >= 10 else major_step
    minor_step = max(1, major_step // 5)

    # Deux dernières majeures en rouge
    last_major_1 = (speed_max // major_step) * major_step
    last_major_2 = last_major_1 - major_step
    red_values = {last_major_1, last_major_2}
    RED = (255, 0, 0)

    def value_to_angle(val: float) -> float:
        frac = (val - speed_min) / span
        frac = max(0.0, min(1.0, frac))
        return 180.0 + 180.0 * frac

    tick_len_major = radius * 0.16
    tick_len_minor = radius * 0.09
    tick_w_major = 3
    tick_w_minor = 2

    # Mesure texte approx
    try:
        tb0 = draw.textbbox((0, 0), "000", font=font)
        approx_label_w = tb0[2] - tb0[0]
        approx_label_h = tb0[3] - tb0[1]
    except Exception:
        approx_label_w, approx_label_h = 20, 12

    # Espacement minimal (en px) → saut régulier
    def px_to_theta(px: float) -> float:
        return px / max(radius, 1.0)

    min_px_minor = 6.0
    min_px_major_tick = 10.0
    min_px_label = max(approx_label_w * 1.1, 18.0)

    theta_min_minor = px_to_theta(min_px_minor)
    theta_min_major_tick = px_to_theta(min_px_major_tick)
    theta_min_label = px_to_theta(min_px_label)

    theta_per_minor = math.pi * (minor_step / span) if span > 0 else math.inf
    theta_per_major = math.pi * (major_step / span) if span > 0 else math.inf

    skip_minor = max(1, int(math.ceil(theta_min_minor / max(theta_per_minor, 1e-9))))
    skip_major_tick = max(1, int(math.ceil(theta_min_major_tick / max(theta_per_major, 1e-9))))
    skip_label = max(1, int(math.ceil(theta_min_label / max(theta_per_major, 1e-9))))

    # --- Ticks ---
    v = math.floor(speed_min / minor_step) * minor_step
    if v < speed_min:
        v += minor_step

    idx_minor = 0
    while v <= speed_max + 1e-6:
        is_major = (v % major_step == 0)
        draw_this = False
        if is_major:
            if v in red_values:
                draw_this = True
            else:
                draw_this = (int((v - speed_min) / major_step + 0.5) % skip_major_tick == 0)
        else:
            draw_this = (idx_minor % skip_minor == 0)

        if draw_this:
            ang = math.radians(value_to_angle(v))
            L = tick_len_major if is_major else tick_len_minor
            ww = tick_w_major if is_major else tick_w_minor
            col = RED if (is_major and v in red_values) else text_color

            x1 = cx + (radius - L) * math.cos(ang)
            y1 = cy + (radius - L) * math.sin(ang)
            x2 = cx + (radius - 2) * math.cos(ang)
            y2 = cy + (radius - 2) * math.sin(ang)
            draw.line([(x1, y1), (x2, y2)], fill=col, width=ww)

        idx_minor += 1
        v += minor_step

    # --- Libellés majeurs (droits) ---
    v = math.floor(speed_min / major_step) * major_step
    if v < speed_min:
        v += major_step

    idx_major = 0
    while v <= speed_max + 1e-6:
        force_show = (v in red_values)
        if force_show or (idx_major % skip_label == 0):
            ang_deg = value_to_angle(v)
            ang = math.radians(ang_deg)
            label = f"{int(v)}"

            try:
                tb = draw.textbbox((0, 0), label, font=font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
            except Exception:
                tw, th = approx_label_w, approx_label_h

            # Légèrement plus à l'extérieur pour ne pas toucher le tick
            r_text = radius - tick_len_major - th * 0.35
            tx = cx + r_text * math.cos(ang) - tw / 2
            ty = cy + r_text * math.sin(ang) - th / 2
            col = RED if force_show else text_color

            if ROTATE_LABELS:
                # (Optionnel) suivre la courbure comme avant
                try:
                    from PIL import Image, ImageDraw as PILImageDraw  # type: ignore
                    canv_w = int(tw * 2)
                    canv_h = int(th * 2)
                    tmp = Image.new("RGBA", (canv_w, canv_h), (0, 0, 0, 0))
                    td = PILImageDraw.Draw(tmp)
                    td.text(((canv_w - tw) / 2, (canv_h - th) / 2), label, font=font, fill=col)
                    rot = ang_deg + 90.0
                    tmp = tmp.rotate(rot, resample=Image.BICUBIC, expand=True)
                    paste_x = int(tx - (tmp.width - tw) / 2)
                    paste_y = int(ty - (tmp.height - th) / 2)
                    base = getattr(draw, "_image", None) or getattr(draw, "im", None)
                    if base is not None and hasattr(base, "paste"):
                        base.paste(tmp, (paste_x, paste_y), tmp)
                    else:
                        draw.text((tx, ty), label, font=font, fill=col)
                except Exception:
                    draw.text((tx, ty), label, font=font, fill=col)
            else:
                # DROIT (horizontal)
                draw.text((tx, ty), label, font=font, fill=col)

        idx_major += 1
        v += major_step
    # --- Affichage du texte de la vitesse courante ---
    speed_text = f"{int(speed_clamped)}"
    try:
        tb = draw.textbbox((0, 0), speed_text, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
    except Exception:
        tw, th = 30, 16

    # Position : centré sur le pivot, mais au-dessus
    tx = cx - tw / 2
    ty = cy - radius * 0.25 - th / 2  # "0.25" règle la hauteur relative

    draw.text((tx, ty), speed_text, font=font, fill=text_color)

    # --- Aiguille + pivot ---
    ang = math.radians(value_to_angle(speed_clamped))
    needle_r = radius - 2
    nx = cx + needle_r * math.cos(ang)
    ny = cy + needle_r * math.sin(ang)
    draw.line([(cx, cy), (nx, ny)], fill=RED, width=3)
    pivot_r = max(3, int(radius * 0.04))
    draw.ellipse([cx - pivot_r, cy - pivot_r, cx + pivot_r, cy + pivot_r], fill=(255, 255, 255))

def draw_info_text(draw, speed, altitude, slope, current_time, draw_area, font, tz, text_color):
    display_time = current_time.astimezone(tz).strftime("%H:%M:%S")
    draw.text((draw_area["x"], draw_area["y"]), f"Vitesse : {speed:.0f} km/h", font=font, fill=text_color)
    draw.text((draw_area["x"], draw_area["y"] + FONT_SIZE_LARGE + 10), f"Altitude : {altitude:.0f} m", font=font, fill=text_color)
    draw.text((draw_area["x"], draw_area["y"] + 2 * (FONT_SIZE_LARGE + 10)), f"Heure : {display_time}", font=font, fill=text_color)
    draw.text((draw_area["x"], draw_area["y"] + 3 * (FONT_SIZE_LARGE + 10)), f"Pente : {slope:.1f} %", font=font, fill=text_color)

# --- AJOUT : texte allure & FC (dessiné sous les 3 lignes existantes) ---
def draw_pace_hr_text(draw, pace_minpk, hr_bpm, draw_area, font, text_color):
    y0 = draw_area["y"] + 4 * (FONT_SIZE_LARGE + 10)
    pace_txt = format_pace_mmss(pace_minpk)
    hr_txt = "—" if hr_bpm is None or not np.isfinite(hr_bpm) else f"{hr_bpm:.0f} bpm"
    draw.text((draw_area["x"], y0), f"Allure : {pace_txt}", font=font, fill=text_color)
    draw.text((draw_area["x"], y0 + (FONT_SIZE_LARGE + 10)), f"FC : {hr_txt}", font=font, fill=text_color)

def draw_north_arrow(img, map_area, rotation_deg, color):
    size = 40
    arrow_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    adraw = ImageDraw.Draw(arrow_img)
    adraw.ellipse((0, 0, size - 1, size - 1), outline=color, width=3)
    adraw.polygon(
        [(size / 2, size * 0.2), (size * 0.35, size * 0.6), (size * 0.65, size * 0.6)],
        fill=color,
    )
    arrow_img = arrow_img.rotate(
        -rotation_deg, resample=Image.BICUBIC, center=(size / 2, size / 2)
    )
    pos_x = map_area["x"] + map_area["width"] - size - 10
    pos_y = map_area["y"] + 10
    img.paste(arrow_img, (int(pos_x), int(pos_y)), arrow_img)

def draw_compass_tape(draw, heading_deg: float, area: dict, font,
                      text_color,
                      bg_color=None,
                      tick_color=(230, 230, 230),
                      center_color=(255, 80, 80),
                      stretch_x: float = 2.8,      # étirement horizontal (plus large)
                      gap_above: int = -220,          # espace entre l'arc et le trait rouge
                      marker_len: int = 70         # longueur du trait rouge
                      ):
    """
    Demi-rose tournée 90° anti-horaire, TEXTES HORIZONTAUX.
    - Ellipse élargie via stretch_x (>1).
    - Ticks: 1°, 5°, 15° (labels tous les 15°).
    - Marqueur: TRAIT ROUGE vertical juste au-dessus de l'arc (pas de label).
    """
    import math

    # ---- Réglages visuels ----
    total_span_deg = 120.0
    arc_span_deg   = 120.0
    arc_width      = 6
    small_len      = 8
    mid_len        = 14
    big_len        = 22
    label_gap      = 8

    # ---- Zone ----
    x, y, w, h = area["x"], area["y"], area["width"], area["height"]
    if w <= 0 or h <= 0:
        return

    # Centre sous la zone pour courbure vers le haut
    cx = x + w / 2.0
    cy = y + h + 6

    # Rayon de base puis ellipse (rx, ry)
    r_base = min(w / 2.0 - 4, h * 1.15)
    if r_base <= 8:
        return
    rx = r_base * stretch_x
    ry = r_base

    # Rotation géométrique globale (CCW)
    ROT = -90.0

    # Arc en coordonnées PIL
    arc_start = 180.0 - arc_span_deg / 2.0
    arc_end   = 180.0 + arc_span_deg / 2.0
    arc_start_r = arc_start + ROT
    arc_end_r   = arc_end   + ROT

    # BBox de l'ellipse et dessin de l'arc
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.arc(bbox, start=arc_start_r, end=arc_end_r, fill=tick_color, width=arc_width)

    # Utilitaires
    def _norm(a):
        a = float(a) % 360.0
        return a if a >= 0 else a + 360.0

    def label_for(d_int: int) -> str:
        d_int = int(_norm(d_int))
        return {0: "N", 90: "E", 180: "S", 270: "O"}.get(d_int, f"{d_int}")

    hdg = _norm(heading_deg)

    # map degré -> angle logique (valeurs plus grandes à GAUCHE)
    def deg_to_theta(d):
        delta = ((d - hdg + 540) % 360) - 180
        if abs(delta) > (total_span_deg / 2.0) + 0.5:
            return None
        k = (delta + total_span_deg / 2.0) / total_span_deg
        k = 1.0 - k
        return arc_start + k * (arc_end - arc_start)

    # coordonnées sur l'ellipse (en tenant compte de la rotation d'affichage)
    def xy_on_ellipse(theta_deg, radius_add=0.0):
        theta_r = math.radians(theta_deg + ROT)
        rx_eff = rx + radius_add * (rx / r_base)
        ry_eff = ry + radius_add * (ry / r_base)
        return (cx + rx_eff * math.cos(theta_r),
                cy + ry_eff * math.sin(theta_r))

    # --- Ticks + labels 15° ---
    start_d = int(math.floor(hdg - total_span_deg / 2.0)) - 1
    end_d   = int(math.ceil(hdg + total_span_deg / 2.0)) + 1

    for d in range(start_d, end_d + 1):
        theta = deg_to_theta(d)
        if theta is None:
            continue
        dd = int(_norm(d))

        if dd % 15 == 0:
            tlen = big_len; width = 2
        elif dd % 5 == 0:
            tlen = mid_len; width = 2
        else:
            tlen = small_len; width = 1

        # graduation
        x1, y1 = xy_on_ellipse(theta, radius_add = -arc_width * 0.5)
        x2, y2 = xy_on_ellipse(theta, radius_add = -arc_width * 0.5 + tlen)
        draw.line([(x1, y1), (x2, y2)], fill=tick_color, width=width)

        # label 15° (texte horizontal)
        if dd % 15 == 0:
            lab = label_for(dd)
            tx, ty = xy_on_ellipse(theta, radius_add = -arc_width * 0.5 + tlen + label_gap)
            tb = draw.textbbox((0, 0), lab, font=font)
            lw, lh = tb[2] - tb[0], tb[3] - tb[1]
            draw.text((tx - lw / 2, ty - lh / 2), lab, font=font, fill=text_color)

    # --- TRAIT ROUGE vertical juste au-dessus de l'arc ---
    # Bord supérieur de l'ellipse + demi-épaisseur de trait de l'arc
    y_top_arc_outer = (cy - ry) - arc_width * 0.5
    y1 = y_top_arc_outer - gap_above           # point le plus proche de l'arc
    y2 = y1 - marker_len                       # vers le haut
    draw.line([(cx, y2), (cx, y1)], fill=center_color, width=4)

def generate_preview_image(resolution, font_path, element_configs, color_configs=None) -> Image.Image:
    bg_c = (color_configs.get("background", rgb_to_hex(BG_COLOR)) if color_configs else rgb_to_hex(BG_COLOR))
    grid_color = (50, 50, 50)
    img = Image.new("RGB", resolution, bg_c)
    draw = ImageDraw.Draw(img)
    try:
        font_display = ImageFont.truetype(font_path, 16)
    except IOError:
        font_display = ImageFont.load_default()
    grid_spacing = 50
    width, height = resolution
    for x_grid in range(0, width, grid_spacing):
        draw.line([(x_grid, 0), (x_grid, height)], fill=grid_color, width=1)
        draw.text((x_grid + 5, 5), str(x_grid), font=font_display, fill=grid_color)
    for y_grid in range(0, height, grid_spacing):
        draw.line([(0, y_grid), (width, y_grid)], fill=grid_color, width=1)
        draw.text((5, y_grid + 5), str(y_grid), font=font_display, fill=grid_color)
    for element_name, config in element_configs.items():
        if config.get("visible"):
            x, y, w, h = config["x"], config["y"], config["width"], config["height"]
            draw.rectangle([x, y, x + w, y + h], outline=(0, 150, 255), width=3)
            try:
                text_bbox = draw.textbbox((0, 0), element_name, font=font_display)
                text_x = x + (w - (text_bbox[2] - text_bbox[0])) // 2
                text_y = y + (h - (text_bbox[3] - text_bbox[1])) // 2
                draw.text((text_x, text_y), element_name, font=font_display, fill=(255, 255, 0))
            except Exception:
                pass
    return img

# ---------- Génération vidéo (zoom 1..12 avec emprise fiable) ----------

def generate_gpx_video(
    gpx_filename,
    output_filename,
    start_offset,
    clip_duration,
    fps,
    resolution,
    font_path,
    element_configs,
    color_configs=None,
    map_style: str = "CyclOSM (FR)",
    zoom_level_ui: int = 8,      # 1..12 (8 = ajusté)
    smoothing_seconds: float = DEFAULT_GRAPH_SMOOTHING_SECONDS,
    progress_callback=None,
    pre_roll_seconds: int = 0,
    post_roll_seconds: int = 0,
) -> bool:
    # --- Config interne ---
    PATCH_FACTOR = 2.4
    MAX_LARGE_DIM = 4096
    VERTICAL_BIAS = 0.65

    # Polices & couleurs
    graph_font_size = compute_graph_font_size(FONT_SIZE_MEDIUM)
    try:
        font_large = ImageFont.truetype(font_path, FONT_SIZE_LARGE)
        font_medium = ImageFont.truetype(font_path, FONT_SIZE_MEDIUM)
        font_graph = ImageFont.truetype(font_path, graph_font_size)
    except IOError:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_graph = ImageFont.load_default()

    bg_c = (color_configs.get("background", BG_COLOR) if color_configs else BG_COLOR)
    map_path_c = (color_configs.get("map_path", PATH_COLOR) if color_configs else PATH_COLOR)
    map_current_path_c = (color_configs.get("map_current_path", CURRENT_PATH_COLOR) if color_configs else CURRENT_PATH_COLOR)
    map_current_point_c = (color_configs.get("map_current_point", CURRENT_POINT_COLOR) if color_configs else CURRENT_POINT_COLOR)
    alt_path_c = (color_configs.get("graph_altitude", PATH_COLOR) if color_configs else PATH_COLOR)
    speed_path_c = (color_configs.get("graph_speed", PATH_COLOR) if color_configs else PATH_COLOR)
    pace_path_c = (color_configs.get("graph_pace", PATH_COLOR) if color_configs else PATH_COLOR)
    hr_path_c = (color_configs.get("graph_hr", PATH_COLOR) if color_configs else PATH_COLOR)
    graph_current_point_c = (color_configs.get("graph_current_point", CURRENT_POINT_COLOR) if color_configs else CURRENT_POINT_COLOR)
    text_c = (color_configs.get("text", TEXT_COLOR) if color_configs else TEXT_COLOR)
    gauge_bg_c = (color_configs.get("gauge_background", GAUGE_BG_COLOR) if color_configs else GAUGE_BG_COLOR)
    compass_area = element_configs.get("Boussole (ruban)", {})


    # GPX
    points, gpx_start, _ = parse_gpx(gpx_filename)
    if not points:
        print("Aucun point GPX.")
        return False
    try:
        if gpx_start is None:
            raise ValueError("Horodatage GPX invalide")
        tz = "Europe/Paris"
        clip_start_time = gpx_start + timedelta(seconds=start_offset)
        start_str = clip_start_time.astimezone(pytz.timezone(tz)).replace(tzinfo=None).isoformat()
        (
            filtered_points,
            extended_start_time,
            tz,
            clip_start_offset,
            clip_end_offset,
            total_span_seconds,
        ) = filter_points_by_time(
            points,
            start_str,
            clip_duration,
            tz,
            pre_margin_seconds=pre_roll_seconds,
            post_margin_seconds=post_roll_seconds,
        )
    except ValueError as e:
        print(f"Erreur: {e}")
        return False

    times_seconds = np.array(
        [
            (pt["time"].astimezone(tz) - extended_start_time).total_seconds()
            for pt in filtered_points
        ],
        dtype=float,
    )
    lats = np.array([pt["lat"] for pt in filtered_points], dtype=float)
    lons = np.array([pt["lon"] for pt in filtered_points], dtype=float)
    eles = np.array([pt["ele"] for pt in filtered_points], dtype=float)
    hrs_raw = np.array([ (pt.get("hr") if pt.get("hr") is not None else np.nan) for pt in filtered_points ], dtype=float)



    data = prepare_track_arrays(
        times_seconds,
        lats,
        lons,
        eles,
        hrs_raw,
        total_span_seconds,
        fps,
        smoothing_seconds,
    )

    total_samples = data["total_frames"]
    if total_samples <= 0:
        print("Aucun échantillon exploitable pour la fenêtre demandée.")
        return False

    clip_start_frame = int(round(clip_start_offset * fps))
    clip_start_frame = max(0, min(total_samples - 1, clip_start_frame))

    clip_end_frame_exclusive = int(round(clip_end_offset * fps))
    clip_end_frame_exclusive = max(clip_start_frame + 1, clip_end_frame_exclusive)
    clip_end_frame_exclusive = min(total_samples, clip_end_frame_exclusive)

    frames_to_render = clip_end_frame_exclusive - clip_start_frame
    if frames_to_render <= 0:
        print("Plage de clip invalide après application des marges.")
        return False

    print(f"Génération de {frames_to_render} images…")
    t0 = time.time()
    report_every = max(1, int(fps))  # ~1 update/s

    def _progress(done: int):
        pct = int(done * 100 / frames_to_render) if frames_to_render else 100
        if progress_callback:
            progress_callback(pct)
        elif done == frames_to_render or (done % report_every == 0):
            elapsed = time.time() - t0
            fps_eff = (done / elapsed) if elapsed > 0 else 0.0
            eta = (frames_to_render - done) / fps_eff if fps_eff > 0 else 0.0
            sys.stdout.write(
                f"\rFrames {done}/{frames_to_render}  | {pct}%  | {fps_eff:.1f} fps  | ETA {eta:0.0f}s"
            )
            sys.stdout.flush()

    interp_times = data["interp_times"]
    interp_lats = data["interp_lats"]
    interp_lons = data["interp_lons"]
    interp_eles = data["interp_eles"]
    interp_speeds = data["interp_speeds"]
    interp_slopes = data["interp_slopes"]
    interp_pace = data["interp_pace"]
    interp_hrs = data["interp_hrs"]


    # Zones UI
    map_area = element_configs.get("Carte", {})
    elev_area = element_configs.get("Profil Altitude", {})
    speed_area = element_configs.get("Profil Vitesse", {})
    pace_area  = element_configs.get("Profil Allure", {})
    hr_area    = element_configs.get("Profil Cardio", {})
    gauge_circ_area = element_configs.get("Jauge Vitesse Circulaire", {})
    gauge_lin_area  = element_configs.get("Jauge Vitesse Linéaire", {})
    gauge_cnt_area  = element_configs.get("Compteur de vitesse", {})
    info_area  = element_configs.get("Infos Texte", {})

    mw = int(map_area.get("width", 0))
    mh = int(map_area.get("height", 0))
    if not map_area.get("visible", False) or mw <= 0 or mh <= 0:
        print("Zone carte non visible ou dimensions nulles.")
        return False

    # BBox brute & centre
    lat_min_raw = float(np.min(lats)); lat_max_raw = float(np.max(lats))
    lon_min_raw = float(np.min(lons)); lon_max_raw = float(np.max(lons))
    lat_c = (lat_min_raw + lat_max_raw) * 0.5
    lon_c = (lon_min_raw + lon_max_raw) * 0.5

    # Profils (altitude & vitesse)
    elev_min = float(np.min(interp_eles))
    elev_max = float(np.max(interp_eles))
    elev_tf = GraphTransformer(elev_min, elev_max, elev_area)
    elev_path = [elev_tf.to_xy(i, val, len(interp_eles)) for i, val in enumerate(interp_eles)]

    speed_min_val = float(np.min(interp_speeds))
    speed_max_val = float(np.max(interp_speeds))
    speed_min, speed_max = auto_speed_bounds(interp_speeds)

    speed_tf = GraphTransformer(speed_min, speed_max, speed_area)
    speed_path = [speed_tf.to_xy(i, val, len(interp_speeds)) for i, val in enumerate(interp_speeds)]

    # --- AJOUT : profils Allure & Cardio ---

    pace_min, pace_max = PACE_GRAPH_MIN, PACE_GRAPH_MAX
    pace_tf = GraphTransformer(pace_min, pace_max, pace_area if pace_area else {"x":0,"y":0,"width":1,"height":1})
    pace_path = [
        pace_tf.to_xy(
            i,
            (
                pace_max
                if not np.isfinite(val)
                else max(pace_min, min(float(val), pace_max))
            ),
            len(interp_pace),
        )
        for i, val in enumerate(interp_pace)
    ]


    has_hr = np.isfinite(interp_hrs).sum() >= 1
    if has_hr:
        hr_vals = interp_hrs[np.isfinite(interp_hrs)]
        hr_min, hr_max = float(np.min(hr_vals)), float(np.max(hr_vals))
    else:
        hr_min, hr_max = 0.0, 1.0
    hr_tf = GraphTransformer(hr_min, hr_max if hr_max > hr_min else (hr_min + 1.0), hr_area if hr_area else {"x":0,"y":0,"width":1,"height":1})
    hr_path = [hr_tf.to_xy(i, (val if np.isfinite(val) else hr_min), len(interp_hrs)) for i, val in enumerate(interp_hrs)]

    graph_specs = [
        (elev_area, elev_path, elev_min, elev_max, "Altitude", "m", alt_path_c),
        (speed_area, speed_path, speed_min_val, speed_max_val, "Vitesse", "km/h", speed_path_c),
        (pace_area, pace_path, pace_min, pace_max, "Allure", "min/km", pace_path_c),
    ]
    if has_hr:
        graph_specs.append((hr_area, hr_path, hr_min, hr_max, "FC", "bpm", hr_path_c))
    graph_layers = prepare_graph_layers(
        resolution,
        font_graph,
        text_c,
        graph_current_point_c,
        graph_specs,
    )

    try:

        writer = imageio.get_writer(output_filename, fps=fps, codec="libx264", macro_block_size=1)

        # Calcul du zoom de base (sur grande image), avec offset UI
        est_w = int(min(MAX_LARGE_DIM, mw * 6))
        est_h = int(min(MAX_LARGE_DIM, mh * 6))
        base_zoom = bbox_fit_zoom(est_w, est_h, lon_min_raw, lat_min_raw, lon_max_raw, lat_max_raw, padding_px=20)
        zoom = max(1, min(19, base_zoom + (zoom_level_ui - 8)))

        # Positions "monde" au zoom choisi
        xs_world, ys_world = lonlat_to_pixel_np(interp_lons, interp_lats, zoom)

        # Etendue, marge et image "large"
        x_min = float(np.min(xs_world)); x_max = float(np.max(xs_world))
        y_min = float(np.min(ys_world)); y_max = float(np.max(ys_world))
        track_w = x_max - x_min; track_h = y_max - y_min

        patch_w = int(math.ceil(mw * PATCH_FACTOR))
        patch_h = int(math.ceil(mh * PATCH_FACTOR))
        margin_x = patch_w // 2 + 64
        margin_y = patch_h // 2 + 64
        width_large  = int(min(MAX_LARGE_DIM, max(est_w, track_w + 2 * margin_x)))
        height_large = int(min(MAX_LARGE_DIM, max(est_h, track_h + 2 * margin_y)))

        # Coin haut-gauche de l'image "large" basé sur le centre du fond de carte
        cx, cy = lonlat_to_pixel(lon_c, lat_c, zoom)
        x0_world = cx - width_large / 2.0
        y0_world = cy - height_large / 2.0

        # Fond "large"
        try:
            base_map_img_large = render_base_map(
                width_large, height_large, map_style, zoom, lon_c, lat_c, bg_c, fail_on_tile_error=True
            )
        except Exception as e:
            print(f"Fond carte non dispo (dyn), fond uni utilisé: {e}")
            base_map_img_large = Image.new("RGB", (width_large, height_large), bg_c)

        # Trace dans le repère "large"
        x_full = xs_world - x0_world
        y_full = ys_world - y0_world
        global_xy = np.column_stack((np.rint(x_full).astype(int), np.rint(y_full).astype(int)))
        local_xy_buffer = np.empty_like(global_xy)

        # Tête lissée (pour rotation)
        if total_samples == 0:
            smoothed_angles = np.zeros(0, dtype=float)
        else:
            dx = np.zeros(total_samples, dtype=float)
            dy = np.zeros(total_samples, dtype=float)
            if total_samples > 1:
                dx[:-1] = np.diff(x_full)
                dy[:-1] = np.diff(y_full)
                dx[-1] = x_full[-1] - x_full[-2]
                dy[-1] = y_full[-1] - y_full[-2]
            headings = np.zeros(total_samples, dtype=float)
            non_zero = (np.abs(dx) > 1e-9) | (np.abs(dy) > 1e-9)
            headings[non_zero] = np.arctan2(dx[non_zero], -dy[non_zero])

            complex_raw = np.exp(1j * headings)
            win_sizes = np.clip((15 - interp_speeds).astype(int), 3, 15)
            half_windows = win_sizes // 2
            indices = np.arange(total_samples)
            start_idx = np.maximum(indices - half_windows, 0)
            end_idx = np.minimum(indices + half_windows + 1, total_samples)

            cumulative = np.concatenate(([0.0 + 0.0j], np.cumsum(complex_raw)))
            counts = (end_idx - start_idx).astype(float)
            counts[counts == 0] = 1.0
            averages = (cumulative[end_idx] - cumulative[start_idx]) / counts
            smoothed_angles = np.arctan2(averages.imag, averages.real)

        # Rendu images -> flux vidéo
        last_heading_deg = (
            math.degrees(smoothed_angles[clip_start_frame])
            if total_samples
            else 0.0
        )

        for frame_count, global_idx in enumerate(
            range(clip_start_frame, clip_end_frame_exclusive)
        ):
            frame_img = Image.new("RGB", resolution, bg_c)
            draw = ImageDraw.Draw(frame_img)

            # Patch centré sur le point courant
            xc = float(x_full[global_idx]); yc = float(y_full[global_idx])
            patch_left = int(round(xc - patch_w / 2.0))
            patch_top  = int(round(yc - patch_h / 2.0))
            patch_img = Image.new("RGB", (patch_w, patch_h), bg_c)

            src_left   = max(0, patch_left)
            src_top    = max(0, patch_top)
            src_right  = min(base_map_img_large.width,  patch_left + patch_w)
            src_bottom = min(base_map_img_large.height, patch_top  + patch_h)
            if src_right > src_left and src_bottom > src_top:
                crop = base_map_img_large.crop((src_left, src_top, src_right, src_bottom))
                dest_x = src_left - patch_left
                dest_y = src_top  - patch_top
                patch_img.paste(crop, (dest_x, dest_y))

            # Trace + progression + point (dans le patch)
            pdraw = ImageDraw.Draw(patch_img)
            np.subtract(global_xy[:, 0], patch_left, out=local_xy_buffer[:, 0])
            np.subtract(global_xy[:, 1], patch_top, out=local_xy_buffer[:, 1])
            local_xy_list = [(int(pt[0]), int(pt[1])) for pt in local_xy_buffer]
            pdraw.line(local_xy_list, fill=map_path_c, width=3)
            pdraw.line(local_xy_list[: global_idx + 1], fill=map_current_path_c, width=4)
            cxp, cyp = local_xy_buffer[global_idx]
            r = 6
            pdraw.ellipse((int(cxp - r), int(cyp - r), int(cxp + r), int(cyp + r)), fill=map_current_point_c)

            # Rotation patch (cadre fixe)
            speed_kmh = float(interp_speeds[global_idx])
            desired_heading = math.degrees(smoothed_angles[global_idx])
            if speed_kmh >= 4.0:
                diff = ((desired_heading - last_heading_deg + 180) % 360) - 180
                last_heading_deg += 0.1 * diff
            heading_deg = last_heading_deg
            patch_img = patch_img.rotate(
                heading_deg,
                resample=Image.BICUBIC,
                expand=False,
                center=(patch_w / 2.0, patch_h / 2.0),
            )

            # Recadrage EXACT viewport
            view_left = int(round(patch_w / 2.0 - mw / 2.0))
            view_top  = int(round(patch_h / 2.0 - VERTICAL_BIAS * mh))
            view = patch_img.crop((view_left, view_top, view_left + mw, view_top + mh))

            # Collage final
            frame_img.paste(view, (int(map_area.get("x", 0)), int(map_area.get("y", 0))))
            draw_north_arrow(frame_img, map_area, heading_deg, text_c)

            # Profils & infos
            for layer in graph_layers:
                frame_img.paste(layer["background"], (0, 0), layer["background"])
            for layer in graph_layers:
                draw_graph_progress_overlay(
                draw,
                layer["path"],
                global_idx,
                layer["base_color"],
                layer["point_color"],
                layer["point_size"],
            )

            if gauge_circ_area.get("visible", False):
                draw_circular_speedometer(
                    draw,
                    float(interp_speeds[global_idx]),
                    speed_min,
                    speed_max,
                    gauge_circ_area,
                    font_medium,
                    gauge_bg_c,
                    text_c,
                )
            if gauge_lin_area.get("visible", False):
                draw_linear_speedometer(
                    draw,
                    float(interp_speeds[global_idx]),
                    speed_min,
                    speed_max,
                    gauge_lin_area,
                    font_medium,
                    gauge_bg_c,
                    text_c,
                )
            if gauge_cnt_area.get("visible", False):
                draw_digital_speedometer(
                    draw,
                    float(interp_speeds[global_idx]),
                    speed_min,
                    speed_max,
                    gauge_cnt_area,
                    font_medium,
                    gauge_bg_c,
                    text_c,
                )
            # Boussole : cap courant
            if compass_area.get("visible", False):
                draw_compass_tape(draw, heading_deg, compass_area, font_medium, text_c)

            if info_area.get("visible", False):
                draw_info_text(draw,
                               float(interp_speeds[global_idx]),
                               float(interp_eles[global_idx]),
                               float(interp_slopes[global_idx]),
                               extended_start_time + timedelta(seconds=float(interp_times[global_idx])),
                               info_area, font_medium, tz, text_c)
                # --- Texte Allure & FC supplémentaires ---
                pace_now = float(interp_pace[global_idx])
                hr_now = float(interp_hrs[global_idx]) if np.isfinite(interp_hrs[global_idx]) else None
                draw_pace_hr_text(draw, pace_now, hr_now, info_area, font_medium, text_c)

            writer.append_data(np.array(frame_img))
            _progress(frame_count + 1)


        writer.close()
        print("Vidéo générée avec succès!")
        return True

    except Exception as e:
        print(f"Erreur écriture vidéo: {e}")
        return False




# ---------- UI Tkinter ----------

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser

# ---------- Aperçu "1re frame" fidèle (mêmes paramètres que la vidéo) ----------
def render_first_frame_image(
    gpx_filename: str,
    start_offset: int,
    clip_duration: int,
    fps: int,
    resolution: tuple[int, int],
    font_path: str,
    element_configs: dict,
    color_configs: dict | None = None,
    map_style: str = "CyclOSM (FR)",
    zoom_level_ui: int = 8,
    smoothing_seconds: float = DEFAULT_GRAPH_SMOOTHING_SECONDS,
    pre_roll_seconds: int = 0,
    post_roll_seconds: int = 0,
):
    # --- Constantes identiques à celles de generate_gpx_video ---
    PATCH_FACTOR = 2.4
    MAX_LARGE_DIM = 4096
    VERTICAL_BIAS = 0.65

    graph_font_size = compute_graph_font_size(FONT_SIZE_MEDIUM)
    try:
        font_large = ImageFont.truetype(font_path, FONT_SIZE_LARGE)
        font_medium = ImageFont.truetype(font_path, FONT_SIZE_MEDIUM)
        font_graph = ImageFont.truetype(font_path, graph_font_size)
    except IOError:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_graph = ImageFont.load_default()

    bg_c = (color_configs.get("background", BG_COLOR) if color_configs else BG_COLOR)
    map_path_c = (color_configs.get("map_path", PATH_COLOR) if color_configs else PATH_COLOR)
    map_current_path_c = (color_configs.get("map_current_path", CURRENT_PATH_COLOR) if color_configs else CURRENT_PATH_COLOR)
    map_current_point_c = (color_configs.get("map_current_point", CURRENT_POINT_COLOR) if color_configs else CURRENT_POINT_COLOR)
    alt_path_c = (color_configs.get("graph_altitude", PATH_COLOR) if color_configs else PATH_COLOR)
    speed_path_c = (color_configs.get("graph_speed", PATH_COLOR) if color_configs else PATH_COLOR)
    pace_path_c = (color_configs.get("graph_pace", PATH_COLOR) if color_configs else PATH_COLOR)
    hr_path_c = (color_configs.get("graph_hr", PATH_COLOR) if color_configs else PATH_COLOR)
    graph_current_point_c = (color_configs.get("graph_current_point", CURRENT_POINT_COLOR) if color_configs else CURRENT_POINT_COLOR)
    text_c = (color_configs.get("text", TEXT_COLOR) if color_configs else TEXT_COLOR)
    gauge_bg_c = (color_configs.get("gauge_background", GAUGE_BG_COLOR) if color_configs else GAUGE_BG_COLOR)

    points, gpx_start, _ = parse_gpx(gpx_filename)
    if not points:
        raise ValueError("Aucun point GPX.")
    if gpx_start is None:
        raise ValueError("Horodatage GPX invalide")
    tz_str = "Europe/Paris"
    clip_start_time = gpx_start + timedelta(seconds=start_offset)
    start_str = clip_start_time.astimezone(pytz.timezone(tz_str)).replace(tzinfo=None).isoformat()
    (
        filtered_points,
        extended_start_time,
        tz,
        clip_start_offset,
        clip_end_offset,
        total_span_seconds,
    ) = filter_points_by_time(
        points,
        start_str,
        clip_duration,
        tz_str,
        pre_margin_seconds=pre_roll_seconds,
        post_margin_seconds=post_roll_seconds,
    )

    times = np.array(
        [
            (pt["time"].astimezone(tz) - extended_start_time).total_seconds()
            for pt in filtered_points
        ],
        dtype=float,
    )
    lats = np.array([pt["lat"] for pt in filtered_points], dtype=float)
    lons = np.array([pt["lon"] for pt in filtered_points], dtype=float)
    eles = np.array([pt["ele"] for pt in filtered_points], dtype=float)
    hrs_raw = np.array([ (pt.get("hr") if pt.get("hr") is not None else np.nan) for pt in filtered_points ], dtype=float)

    data = prepare_track_arrays(
        times,
        lats,
        lons,
        eles,
        hrs_raw,
        total_span_seconds,
        fps,
        smoothing_seconds,
    )
    total_frames = data["total_frames"]
    if total_frames <= 0:
        raise ValueError("Aucun échantillon exploitable pour la fenêtre demandée.")

    clip_start_frame = int(round(clip_start_offset * fps))
    clip_start_frame = max(0, min(total_frames - 1, clip_start_frame))

    interp_times = data["interp_times"]
    interp_lats = data["interp_lats"]
    interp_lons = data["interp_lons"]
    interp_eles = data["interp_eles"]
    interp_speeds = data["interp_speeds"]
    interp_slopes = data["interp_slopes"]
    interp_pace = data["interp_pace"]
    interp_hrs = data["interp_hrs"]


    map_area = element_configs.get("Carte", {})
    elev_area = element_configs.get("Profil Altitude", {})
    speed_area = element_configs.get("Profil Vitesse", {})
    pace_area  = element_configs.get("Profil Allure", {})
    hr_area    = element_configs.get("Profil Cardio", {})
    gauge_circ_area = element_configs.get("Jauge Vitesse Circulaire", {})
    gauge_lin_area  = element_configs.get("Jauge Vitesse Linéaire", {})
    gauge_cnt_area  = element_configs.get("Compteur de vitesse", {})
    info_area  = element_configs.get("Infos Texte", {})
    compass_area = element_configs.get("Boussole (ruban)", {})


    mw = int(map_area.get("width", 0)); mh = int(map_area.get("height", 0))
    if not map_area.get("visible", False) or mw <= 0 or mh <= 0:
        raise ValueError("Zone carte non visible ou dimensions nulles.")

    lat_min_raw = float(np.min(lats)); lat_max_raw = float(np.max(lats))
    lon_min_raw = float(np.min(lons)); lon_max_raw = float(np.max(lons))
    lat_c = (lat_min_raw + lat_max_raw) * 0.5
    lon_c = (lon_min_raw + lon_max_raw) * 0.5

    elev_min = float(np.min(interp_eles))
    elev_max = float(np.max(interp_eles))
    elev_tf = GraphTransformer(elev_min, elev_max, elev_area)
    elev_path = [elev_tf.to_xy(i, val, len(interp_eles)) for i, val in enumerate(interp_eles)]

    speed_min_val = float(np.min(interp_speeds))
    speed_max_val = float(np.max(interp_speeds))
    speed_min, speed_max = auto_speed_bounds(interp_speeds)

    speed_tf = GraphTransformer(speed_min, speed_max, speed_area)
    speed_path = [speed_tf.to_xy(i, val, len(interp_speeds)) for i, val in enumerate(interp_speeds)]

    pace_min, pace_max = PACE_GRAPH_MIN, PACE_GRAPH_MAX
    pace_tf = GraphTransformer(pace_min, pace_max, pace_area if pace_area else {"x":0,"y":0,"width":1,"height":1})
    pace_path = [
        pace_tf.to_xy(
            i,
            (
                pace_max
                if not np.isfinite(val)
                else max(pace_min, min(float(val), pace_max))
            ),
            len(interp_pace),
        )
        for i, val in enumerate(interp_pace)
    ]

    has_hr = np.isfinite(interp_hrs).sum() >= 1
    if has_hr:
        hr_vals = interp_hrs[np.isfinite(interp_hrs)]
        hr_min, hr_max = float(np.min(hr_vals)), float(np.max(hr_vals))
    else:
        hr_min, hr_max = 0.0, 1.0
    hr_tf = GraphTransformer(hr_min, hr_max if hr_max > hr_min else (hr_min + 1.0), hr_area if hr_area else {"x":0,"y":0,"width":1,"height":1})
    hr_path = [hr_tf.to_xy(i, (val if np.isfinite(val) else hr_min), len(interp_hrs)) for i, val in enumerate(interp_hrs)]

    graph_specs = [
        (elev_area, elev_path, elev_min, elev_max, "Altitude", "m", alt_path_c),
        (speed_area, speed_path, speed_min_val, speed_max_val, "Vitesse", "km/h", speed_path_c),
        (pace_area, pace_path, pace_min, pace_max, "Allure", "min/km", pace_path_c),
    ]
    if has_hr:
        graph_specs.append((hr_area, hr_path, hr_min, hr_max, "FC", "bpm", hr_path_c))
    graph_layers = prepare_graph_layers(
        resolution,
        font_graph,
        text_c,
        graph_current_point_c,
        graph_specs,
    )

    frame_img = Image.new("RGB", resolution, bg_c)
    draw = ImageDraw.Draw(frame_img)
    current_idx = clip_start_frame
    heading_deg = 0.0

    try:
        est_w = int(min(MAX_LARGE_DIM, mw * 6))
        est_h = int(min(MAX_LARGE_DIM, mh * 6))
        base_zoom = bbox_fit_zoom(est_w, est_h, lon_min_raw, lat_min_raw, lon_max_raw, lat_max_raw, padding_px=20)
        zoom = max(1, min(19, base_zoom + (zoom_level_ui - 8)))

        xs_world, ys_world = lonlat_to_pixel_np(interp_lons, interp_lats, zoom)

        x_min = float(np.min(xs_world)); x_max = float(np.max(xs_world))
        y_min = float(np.min(ys_world)); y_max = float(np.max(ys_world))
        track_w = x_max - x_min; track_h = y_max - y_min


        patch_w = int(math.ceil(mw * PATCH_FACTOR))
        patch_h = int(math.ceil(mh * PATCH_FACTOR))
        margin_x = patch_w // 2 + 64
        margin_y = patch_h // 2 + 64
        width_large = int(min(MAX_LARGE_DIM, max(est_w, track_w + 2 * margin_x)))
        height_large = int(min(MAX_LARGE_DIM, max(est_h, track_h + 2 * margin_y)))

        cx, cy = lonlat_to_pixel(lon_c, lat_c, zoom)
        x0_world = cx - width_large / 2.0
        y0_world = cy - height_large / 2.0

        try:
            base_map_img_large = render_base_map(
                width_large, height_large, map_style, zoom, lon_c, lat_c, bg_c, fail_on_tile_error=True
            )
        except Exception:
            base_map_img_large = Image.new("RGB", (width_large, height_large), bg_c)

        x_full = xs_world - x0_world
        y_full = ys_world - y0_world
        global_xy = np.column_stack((np.rint(x_full).astype(int), np.rint(y_full).astype(int)))
        local_xy_buffer = np.empty_like(global_xy)

        headings = []
        for i in range(total_frames):
            if i < total_frames - 1:
                dx = x_full[i + 1] - x_full[i]
                dy = y_full[i + 1] - y_full[i]
            else:
                dx = x_full[i] - x_full[i - 1]
                dy = y_full[i] - y_full[i - 1]
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                headings.append(math.atan2(dx, -dy))
            else:
                headings.append(0.0)
        complex_raw = np.exp(1j * np.array(headings))
        win_sizes = np.clip((15 - interp_speeds).astype(int), 3, 15)
        smoothed_angles = []
        for idx in range(total_frames):
            w = int(win_sizes[idx])
            half_w = w // 2
            a = max(0, idx - half_w)
            b = min(total_frames, idx + half_w + 1)
            avg = np.mean(complex_raw[a:b])
            smoothed_angles.append(math.atan2(avg.imag, avg.real))

        xc = float(x_full[current_idx])
        yc = float(y_full[current_idx])
        patch_left = int(round(xc - patch_w / 2.0))
        patch_top = int(round(yc - patch_h / 2.0))
        patch_img = Image.new("RGB", (patch_w, patch_h), bg_c)

        src_left = max(0, patch_left)
        src_top = max(0, patch_top)
        src_right = min(base_map_img_large.width, patch_left + patch_w)
        src_bottom = min(base_map_img_large.height, patch_top + patch_h)
        if src_right > src_left and src_bottom > src_top:
            crop = base_map_img_large.crop((src_left, src_top, src_right, src_bottom))
            dest_x = src_left - patch_left
            dest_y = src_top - patch_top
            patch_img.paste(crop, (dest_x, dest_y))

        pdraw = ImageDraw.Draw(patch_img)
        np.subtract(global_xy[:, 0], patch_left, out=local_xy_buffer[:, 0])
        np.subtract(global_xy[:, 1], patch_top, out=local_xy_buffer[:, 1])
        local_xy_list = [(int(pt[0]), int(pt[1])) for pt in local_xy_buffer]
        pdraw.line(local_xy_list, fill=map_path_c, width=3)
        pdraw.line(local_xy_list[: current_idx + 1], fill=map_current_path_c, width=4)
        cxp, cyp = local_xy_buffer[current_idx]
        r = 6
        pdraw.ellipse((int(cxp - r), int(cyp - r), int(cxp + r), int(cyp + r)), fill=map_current_point_c)

        speed_kmh0 = float(interp_speeds[current_idx])
        heading_deg = (
            0.0
            if speed_kmh0 < 4.0
            else math.degrees(smoothed_angles[current_idx])
        )
        patch_img = patch_img.rotate(
            heading_deg,
            resample=Image.BICUBIC,
            expand=False,
            center=(patch_w / 2.0, patch_h / 2.0),
        )
        view_left = int(round(patch_w / 2.0 - mw / 2.0))
        view_top = int(round(patch_h / 2.0 - VERTICAL_BIAS * mh))
        view = patch_img.crop((view_left, view_top, view_left + mw, view_top + mh))

        frame_img.paste(view, (int(map_area.get("x", 0)), int(map_area.get("y", 0))))

        draw_north_arrow(frame_img, map_area, heading_deg, text_c)

    except Exception as e:
        pass

    for layer in graph_layers:
        frame_img.paste(layer["background"], (0, 0), layer["background"])
    for layer in graph_layers:
        draw_graph_progress_overlay(
            draw,
            layer["path"],
            current_idx,
            layer["base_color"],
            layer["point_color"],
            layer["point_size"],
        )

    if gauge_circ_area.get("visible", False):
        draw_circular_speedometer(
            draw,
            float(interp_speeds[current_idx]),
            speed_min,
            speed_max,
            gauge_circ_area,
            font_medium,
            gauge_bg_c,
            text_c,
        )
    if gauge_lin_area.get("visible", False):
        draw_linear_speedometer(
            draw,
            float(interp_speeds[current_idx]),
            speed_min,
            speed_max,
            gauge_lin_area,
            font_medium,
            gauge_bg_c,
            text_c,
        )
    if gauge_cnt_area.get("visible", False):
        draw_digital_speedometer(
            draw,
            float(interp_speeds[current_idx]),
            speed_min,
            speed_max,
            gauge_cnt_area,
            font_medium,
            gauge_bg_c,
            text_c,
        )
    if compass_area.get("visible", False):
        draw_compass_tape(draw, heading_deg, compass_area, font_medium, text_c)
    if info_area.get("visible", False):
        draw_info_text(draw,
                       float(interp_speeds[current_idx]),
                       float(interp_eles[current_idx]),
                       float(interp_slopes[current_idx]),
                       extended_start_time + timedelta(seconds=float(interp_times[current_idx])),
                       info_area, font_medium, tz, text_c)
        pace_now = float(interp_pace[current_idx])
        hr_now = float(interp_hrs[current_idx]) if np.isfinite(interp_hrs[current_idx]) else None
        draw_pace_hr_text(draw, pace_now, hr_now, info_area, font_medium, text_c)

    return frame_img

class GPXVideoApp:
    def __init__(self, master):
        self.master = master
        master.title("Overlay GPX")
        master.geometry("1650x600")
        master.minsize(1200, 600)
        self.accent_button_style = "TButton"
        try:
            style = ttk.Style(); style.theme_use("clam")
            base_bg = "#f4f6fb"
            accent = "#2f6fed"
            style.configure("TFrame", background=base_bg)
            style.configure("TLabel", background=base_bg)
            style.configure("TLabelframe", background=base_bg)
            style.configure("TLabelframe.Label", background=base_bg, foreground="#1f2a44")
            style.configure("Card.TFrame", background="#ffffff", relief="flat")
            style.configure("Toolbar.TFrame", background=base_bg)
            style.configure("Accent.TButton", padding=(10, 6), foreground="#ffffff", background=accent)
            style.map("Accent.TButton", background=[("active", "#244fd2")])
            style.configure("TNotebook", background=base_bg, padding=(4, 4))
            style.configure("TNotebook.Tab", padding=(14, 6))
            style.configure("TButton", padding=(8, 4))
            self.accent_button_style = "Accent.TButton"
        except Exception:
            style = None
        self.master.configure(background="#f4f6fb")

        self.gpx_file_path = ""
        self.gpx_start_time_raw = None
        self.gpx_end_time_raw = None
        self.start_offset_var = tk.IntVar(value=0)
        self.pre_roll_var = tk.IntVar(value=0)
        self.post_roll_var = tk.IntVar(value=0)

        self.element_visibility_vars = {}
        self.element_pos_entries_vars = {}
        self.element_sliders_vars = {}
        self.element_sliders = {}
        self.element_calculated_height_labels = {}
        self.element_initial_ratios = {}

        self.preview_image_tk = None
        self.current_video_resolution = list(DEFAULT_RESOLUTION)
        self._block_recursion = False
        self.resolution_presets = [
            ("1080p (1920x1080)", (1920, 1080)),
            ("4K (3840x2160)", (3840, 2160)),
        ]
        self._resolution_lookup = {label: value for label, value in self.resolution_presets}

        # Couleurs
        self.color_configs = {}
        self.color_preview_frames = {}
        self.default_color_map = {
            "background": BG_COLOR,
            "map_path": PATH_COLOR,
            "map_current_path": CURRENT_PATH_COLOR,
            "map_current_point": CURRENT_POINT_COLOR,
            "graph_altitude": PATH_COLOR,
            "graph_speed": (0, 255, 0),
            "graph_pace": (255, 165, 0),
            "graph_hr": (255, 0, 0),
            "graph_current_point": CURRENT_POINT_COLOR,
            "text": TEXT_COLOR,
            "gauge_background": GAUGE_BG_COLOR,
        }
        self._initialize_color_configs()

        # Police d'écriture
        self.font_path_var = tk.StringVar(value=DEFAULT_FONT_PATH)
        self.font_size_var = tk.IntVar(value=FONT_SIZE_LARGE)
        self.graph_smoothing_seconds_var = tk.StringVar(
            value=str(int(DEFAULT_GRAPH_SMOOTHING_SECONDS))
        )

        # Validation entrées
        self.vcmd_int = (master.register(self.validate_integer_or_empty), "%P")
        self.vcmd_float = (master.register(self.validate_float_or_empty), "%P")

        # Style de carte + Zoom (1..12)
        self.map_style_var = tk.StringVar(value="CyclOSM (FR)")
        self.map_zoom_level_var = tk.IntVar(value=8)  # 1..12

        # Labels conviviaux
        self.color_labels = {
            "background": "Fond",
            "map_path": "Tracé carte (futur)",
            "map_current_path": "Tracé carte (passé)",
            "map_current_point": "Point carte",
            "graph_altitude": "Profil altitude",
            "graph_speed": "Profil vitesse",
            "graph_pace": "Profil allure",
            "graph_hr": "Profil FC",
            "graph_current_point": "Point profil",
            "text": "Texte",
            "gauge_background": "Fond jauge",
        }
        self.progress_message_default = "Temps restant estimé : --:--:--"
        self.generate_btn = None
        self.create_widgets()
        self.populate_initial_element_ratios()
        self.update_preview_area_size()
        self.update_all_slider_ranges()
        self.show_preview(initial_load=True)
        self.update_pre_roll_label()
        self.update_post_roll_label()
        self.update_display_time_label()
        self.update_margin_scales()

    def _initialize_color_configs(self):
        for key, rgb_tuple in self.default_color_map.items():
            self.color_configs[key] = rgb_to_hex(rgb_tuple)
        for key, frame in self.color_preview_frames.items():
            frame.config(background=self.color_configs.get(key, "#FFFFFF"))

    def populate_initial_element_ratios(self):
        for element_name, defaults in DEFAULT_ELEMENT_CONFIGS.items():
            if defaults["width"] > 0:
                if defaults["height"] >= 0:
                    self.element_initial_ratios[element_name] = defaults["height"] / defaults["width"]
                else:
                    self.element_initial_ratios[element_name] = 1.0
            elif defaults["height"] > 0:
                self.element_initial_ratios[element_name] = float("inf")
            else:
                self.element_initial_ratios[element_name] = 1.0
            if element_name in self.element_calculated_height_labels:
                w = defaults["width"]; ratio = self.element_initial_ratios[element_name]
                h = int(round(w * ratio)) if ratio != float("inf") else defaults["height"]
                self.element_calculated_height_labels[element_name].set(str(h))

    def validate_integer_or_empty(self, value_if_allowed):
        if value_if_allowed == "" or value_if_allowed == "-":
            return True
        try:
            int(value_if_allowed); return True
        except ValueError:
            return False

    def validate_float_or_empty(self, value_if_allowed):
        if value_if_allowed in ("", "-", ".", "-.", ",", "-,"):
            return True
        try:
            float(value_if_allowed.replace(",", "."))
            return True
        except ValueError:
            return False

    # ----- UI -----
    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding=(10, 10))
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Barre d’outils
        toolbar = ttk.Frame(main_frame, style="Toolbar.TFrame")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for col in range(6):
            toolbar.columnconfigure(col, weight=0)

        ttk.Button(toolbar, text="Ouvrir GPX", command=self.select_gpx_file).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(toolbar, text="Enr. preset", command=self.save_configuration_preset).grid(row=0, column=1, padx=3)
        ttk.Button(toolbar, text="Charger preset", command=self.load_configuration_preset).grid(row=0, column=2, padx=3)
        ttk.Button(toolbar, text="Prévisualiser 1ʳᵉ frame", command=self.preview_first_frame).grid(row=0, column=3, padx=6)
        self.generate_btn = ttk.Button(toolbar, text="Générer Vidéo", style=self.accent_button_style, command=self.generate_video)
        self.generate_btn.grid(row=0, column=4, sticky="w", padx=6)
        ttk.Button(toolbar, text="Aide", command=self.show_help).grid(row=0, column=5, padx=(12, 0))
        ttk.Separator(toolbar, orient=tk.VERTICAL).grid(row=0, column=6, sticky="ns", padx=12)
        self.gpx_toolbar_label_var = tk.StringVar(value="GPX: aucun")
        ttk.Label(toolbar, textvariable=self.gpx_toolbar_label_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=7, sticky="w")
        toolbar.columnconfigure(7, weight=1)

        self.progress_time_var = tk.StringVar(value=self.progress_message_default)
        ttk.Label(toolbar, textvariable=self.progress_time_var, anchor="e").grid(row=0, column=8, sticky="e", padx=(12, 0))

        # Colonne gauche avec onglets pour condenser l'interface
        config_panel_outer = ttk.Notebook(main_frame); self.config_panel_outer = config_panel_outer
        config_panel_outer.grid(row=1, column=0, sticky="nsw", padx=(0, 12))

        gen_params_tab = ttk.Frame(config_panel_outer)
        config_panel_outer.add(gen_params_tab, text="Paramètres")
        gen_params_frame = ttk.LabelFrame(gen_params_tab, text="Paramètres de génération", padding=(8, 6))
        gen_params_frame.pack(fill=tk.BOTH, expand=True, pady=4, padx=6)
        for col in range(3):
            gen_params_frame.columnconfigure(col, weight=1 if col == 1 else 0)

        row = 0
        ttk.Label(gen_params_frame, text="Fichier GPX:").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=(0, 4))
        self.gpx_label = ttk.Label(gen_params_frame, text="Aucun", anchor="w", wraplength=220)
        self.gpx_label.grid(row=row, column=1, columnspan=2, sticky="ew", pady=(0, 4))

        row += 1
        info_frame = ttk.Frame(gen_params_frame, style="Card.TFrame", padding=(8, 6))
        info_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        for col in range(3):
            info_frame.columnconfigure(col, weight=1)
        self.gpx_start_time_label = ttk.Label(info_frame, text="Début GPX: N/A", anchor="w")
        self.gpx_start_time_label.grid(row=0, column=0, sticky="w")
        self.gpx_end_time_label = ttk.Label(info_frame, text="Fin GPX: N/A", anchor="center")
        self.gpx_end_time_label.grid(row=0, column=1, sticky="ew")
        self.gpx_duration_label = ttk.Label(info_frame, text="Durée GPX: N/A", anchor="e")
        self.gpx_duration_label.grid(row=0, column=2, sticky="e")

        def place_slider_row(label_text, scale_attr_name, value_attr_name):
            nonlocal row
            row += 1
            ttk.Label(gen_params_frame, text=label_text).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            scale_widget = getattr(self, scale_attr_name, None)
            if scale_widget is None:
                raise AttributeError(f"Missing attribute {scale_attr_name}")
            scale_widget.grid(row=row, column=1, sticky="ew", pady=2)
            value_label = getattr(self, value_attr_name, None)
            if value_label is None:
                raise AttributeError(f"Missing attribute {value_attr_name}")
            value_label.grid(row=row, column=2, sticky="e", padx=(8, 0))

        self.start_offset_scale = ttk.Scale(gen_params_frame, from_=0, to=0, variable=self.start_offset_var, orient=tk.HORIZONTAL)
        self.start_offset_label = ttk.Label(gen_params_frame, text="0:00:00", anchor="e")
        self.start_offset_var.trace_add("write", self.on_start_offset_change)
        self.update_start_offset_label()
        place_slider_row("Début du clip:", "start_offset_scale", "start_offset_label")

        self.duration_var = tk.IntVar(value=DEFAULT_CLIP_DURATION_SECONDS)
        self.duration_scale = ttk.Scale(gen_params_frame, from_=1, to=1, variable=self.duration_var, orient=tk.HORIZONTAL)
        self.duration_label = ttk.Label(gen_params_frame, text="0:00:00", anchor="e")
        place_slider_row("Durée du clip:", "duration_scale", "duration_label")

        row += 1
        self.clip_time_label = ttk.Label(gen_params_frame, text="Clip: début N/A - fin N/A")
        self.clip_time_label.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        self.pre_roll_scale = ttk.Scale(gen_params_frame, from_=0, to=0, variable=self.pre_roll_var, orient=tk.HORIZONTAL)
        self.pre_roll_label = ttk.Label(gen_params_frame, text="0 s", anchor="e")
        self.pre_roll_var.trace_add("write", self.on_pre_roll_change)
        row += 1
        ttk.Label(gen_params_frame, text="Temps affiché avant le clip (s):").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        self.pre_roll_scale.grid(row=row, column=1, sticky="ew", pady=2)
        self.pre_roll_label.grid(row=row, column=2, sticky="e", padx=(8, 0))

        self.post_roll_scale = ttk.Scale(gen_params_frame, from_=0, to=0, variable=self.post_roll_var, orient=tk.HORIZONTAL)
        self.post_roll_label = ttk.Label(gen_params_frame, text="0 s", anchor="e")
        self.post_roll_var.trace_add("write", self.on_post_roll_change)
        row += 1
        ttk.Label(gen_params_frame, text="Temps affiché après le clip (s):").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        self.post_roll_scale.grid(row=row, column=1, sticky="ew", pady=2)
        self.post_roll_label.grid(row=row, column=2, sticky="e", padx=(8, 0))

        row += 1
        self.display_time_label = ttk.Label(gen_params_frame, text="Fenêtre affichée: début N/A - fin N/A")
        self.display_time_label.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.duration_var.trace_add("write", self.on_duration_slider_change)
        self.update_duration_label()
        self.update_clip_time_label()

        row += 1
        ttk.Label(gen_params_frame, text="FPS:").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        self.fps_entry_var = tk.StringVar(value=str(DEFAULT_FPS))
        self.fps_entry = ttk.Entry(gen_params_frame, textvariable=self.fps_entry_var, validate="key", validatecommand=self.vcmd_int)
        self.fps_entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)

        row += 1
        ttk.Label(gen_params_frame, text="Intervalle de lissage des données (s):").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        self.graph_smoothing_entry = ttk.Entry(
            gen_params_frame,
            textvariable=self.graph_smoothing_seconds_var,
            validate="key",
            validatecommand=self.vcmd_float,
        )
        self.graph_smoothing_entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)

        row += 1
        ttk.Label(gen_params_frame, text="Résolution Vidéo :").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        resolution_labels = [label for label, _ in self.resolution_presets]
        default_label = next(
            (label for label, res in self.resolution_presets if res == tuple(self.current_video_resolution)),
            resolution_labels[0],
        )
        self.resolution_choice_var = tk.StringVar(value=default_label)
        self.resolution_combo = ttk.Combobox(
            gen_params_frame,
            textvariable=self.resolution_choice_var,
            values=resolution_labels,
            state="readonly",
        )
        self.resolution_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        self.resolution_combo.bind("<<ComboboxSelected>>", self.on_resolution_selection_change)
        row += 1
        ttk.Label(gen_params_frame, text="Police (TTF):").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        font_frame = ttk.Frame(gen_params_frame)
        font_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        font_frame.columnconfigure(0, weight=1)
        ttk.Entry(font_frame, textvariable=self.font_path_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(font_frame, text="Parcourir", command=self.select_font_file).grid(row=0, column=1, padx=(6, 0))

        row += 1
        ttk.Label(gen_params_frame, text="Taille police:").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(gen_params_frame, textvariable=self.font_size_var, validate="key", validatecommand=self.vcmd_int).grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)

        # Style de carte
        row += 1
        ttk.Label(gen_params_frame, text="Style de carte:").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        style_choices = list(MAP_TILE_SERVERS.keys())
        self.map_style_combo = ttk.Combobox(gen_params_frame, textvariable=self.map_style_var, values=style_choices, state="readonly")
        self.map_style_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)

        # Zoom 1..12 (combobox) — 8 = “ajusté”
        row += 1
        ttk.Label(gen_params_frame, text="Zoom (1-Loin à 12-Proche) :").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=(6, 2))
        self.zoom_combo = ttk.Combobox(gen_params_frame, textvariable=self.map_zoom_level_var, state="readonly",
                                       values=[str(i) for i in range(1, 13)])
        self.zoom_combo.set(str(self.map_zoom_level_var.get()))
        self.zoom_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        # Disposition des éléments (onglet dédié)
        elements_tab = ttk.Frame(config_panel_outer)
        config_panel_outer.add(elements_tab, text="Disposition")
        elements_outer_frame = ttk.LabelFrame(elements_tab, text="Disposition des éléments", padding=(6, 6))
        elements_outer_frame.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
        scrollable_frame_elements = ttk.Frame(elements_outer_frame)
        scrollable_frame_elements.pack(fill=tk.BOTH, expand=True)

        headers = ["Élément", "Aff.", "X", "Y", "Largeur"]
        for i, _ in enumerate(headers):
            scrollable_frame_elements.columnconfigure(i, weight=1 if i in [0, 2, 3, 4] else 0, minsize=50 if i != 0 else 80)
        for col, header_text in enumerate(headers):
            ttk.Label(scrollable_frame_elements, text=header_text).grid(row=0, column=col, padx=2, pady=2, sticky="w" if col == 0 else "nsew")

        row_num = 1
        for element_name, defaults in DEFAULT_ELEMENT_CONFIGS.items():
            ttk.Label(scrollable_frame_elements, text=element_name, wraplength=70).grid(row=row_num, column=0, padx=5, pady=2, sticky="w")
            var = tk.BooleanVar(value=defaults["visible"])
            cb_visible = ttk.Checkbutton(scrollable_frame_elements, variable=var, command=lambda el=element_name: self.handle_element_change(el, "visible"))
            cb_visible.grid(row=row_num, column=1, padx=2, pady=2)
            self.element_visibility_vars[element_name] = var

            self.element_pos_entries_vars[element_name] = {}
            self.element_sliders_vars[element_name] = {}
            self.element_sliders[element_name] = {}
            self.element_calculated_height_labels[element_name] = tk.StringVar()

            for key_idx, key in enumerate(["x", "y", "width"]):
                current_col = key_idx + 2
                entry_slider_frame = ttk.Frame(scrollable_frame_elements)
                entry_slider_frame.grid(row=row_num, column=current_col, padx=1, pady=1, sticky="ew")
                entry_slider_frame.columnconfigure(0, weight=1)
                entry_slider_frame.columnconfigure(1, weight=0)

                entry_var = tk.StringVar(value=str(defaults[key])); self.element_pos_entries_vars[element_name][key] = entry_var
                slider_var = tk.IntVar(value=defaults[key]); self.element_sliders_vars[element_name][key] = slider_var

                entry_widget = ttk.Entry(entry_slider_frame, textvariable=entry_var, width=4, validate="key", validatecommand=self.vcmd_int)
                entry_widget.grid(row=0, column=1, sticky="e")
                entry_widget.bind("<FocusOut>", lambda e, el=element_name, k=key: self.handle_element_change(el, k, "entry"))
                entry_widget.bind("<Return>",  lambda e, el=element_name, k=key: self.handle_element_change(el, k, "entry"))

                slider = ttk.Scale(entry_slider_frame, variable=slider_var, orient=tk.HORIZONTAL, length=150,
                                   command=lambda val, el=element_name, k=key: self.handle_element_change(el, k, "slider"))
                slider.grid(row=0, column=0, sticky="ew", padx=(0, 2))
                self.element_sliders[element_name][key] = slider

            row_num += 1
        # Onglet Couleurs
        colors_tab = ttk.Frame(config_panel_outer)
        config_panel_outer.add(colors_tab, text="Couleurs")
        colors_outer = ttk.LabelFrame(colors_tab, text="Palette de couleurs", padding=(6, 6))
        colors_outer.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
        self.color_preview_frames = {}
        row = 0
        for key, label_text in self.color_labels.items():
            ttk.Label(colors_outer, text=label_text + " :").grid(row=row, column=0, sticky="w", padx=4, pady=4)
            swatch = tk.Frame(colors_outer, width=28, height=18, relief="sunken", borderwidth=1,
                              background=self.color_configs.get(key, "#FFFFFF"))
            swatch.grid(row=row, column=1, padx=6, pady=4, sticky="w")
            self.color_preview_frames[key] = swatch
            def make_pick(k=key, frame=swatch):
                return lambda: self.pick_color(k, frame)
            ttk.Button(colors_outer, text="Modifier…", command=make_pick()).grid(row=row, column=2, padx=4, pady=4)
            row += 1
        btns = ttk.Frame(colors_outer)
        btns.grid(row=row, column=0, columnspan=3, pady=(4, 0))
        def reset_colors():
            for k, rgb in self.default_color_map.items():
                self.color_configs[k] = rgb_to_hex(rgb)
                if k in self.color_preview_frames:
                    self.color_preview_frames[k].config(background=self.color_configs[k])
            self.show_preview(force_update=True)
        ttk.Button(btns, text="Réinitialiser", command=reset_colors).pack(side=tk.LEFT)

        # Panneau d’aperçu
        preview_panel_frame = ttk.LabelFrame(main_frame, text="Aperçu de la disposition", padding=(6, 6))
        preview_panel_frame.grid(row=1, column=1, sticky="nsew")
        self.preview_label = ttk.Label(preview_panel_frame, text="L'aperçu apparaîtra ici.", anchor="center", relief="groove")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.preview_label.bind("<Configure>", self.on_preview_resize)

    def show_help(self):
        """Affiche une fenêtre d’aide avec un résumé des fonctionnalités."""

        help_sections = [
            "Ouvrir GPX : sélectionne le fichier GPX à analyser et met à jour toutes les données disponibles.",
            "Prévisualiser 1ʳᵉ frame : génère un aperçu statique pour vérifier la mise en page et les couleurs.",
            "Générer Vidéo : lance le rendu complet de la vidéo avec les paramètres courants.",
            "Paramètres : ajustez la plage temporelle, la durée du clip, les pré/post-roll ainsi que la résolution et la police.",
            "Disposition : cochez les éléments à afficher et modifiez leurs positions/largeurs en direct.",
            "Couleurs : personnalisez chaque couleur de l’overlay et réinitialisez rapidement si nécessaire.",
        ]
        messagebox.showinfo("Aide — Overlay GPX", "\n\n".join(help_sections))

    def pick_color(self, key, preview_frame=None):
        initial = self.color_configs.get(key, "#FFFFFF")
        color_code = colorchooser.askcolor(color=initial, title=f"Choisir une couleur — {self.color_labels.get(key, key)}")[1]
        if color_code:
            self.color_configs[key] = color_code
            if preview_frame is not None: preview_frame.config(background=color_code)
            self.show_preview(force_update=True)

    def _collect_current_configuration(self) -> dict:
        element_configs = {}
        for name in DEFAULT_ELEMENT_CONFIGS.keys():
            height_var = self.element_calculated_height_labels.get(name)
            try:
                height_val = int(height_var.get()) if height_var and height_var.get() else DEFAULT_ELEMENT_CONFIGS[name]["height"]
            except (ValueError, tk.TclError):
                height_val = DEFAULT_ELEMENT_CONFIGS[name]["height"]

            visibility_var = self.element_visibility_vars.get(name)
            visible = bool(visibility_var.get()) if visibility_var is not None else DEFAULT_ELEMENT_CONFIGS[name]["visible"]

            def _safe_entry_value(key: str) -> int:
                entry_var = self.element_pos_entries_vars.get(name, {}).get(key)
                if entry_var is None:
                    return DEFAULT_ELEMENT_CONFIGS[name][key]
                try:
                    return int(entry_var.get())
                except (ValueError, tk.TclError):
                    return DEFAULT_ELEMENT_CONFIGS[name][key]

            element_configs[name] = {
                "visible": visible,
                "x": _safe_entry_value("x"),
                "y": _safe_entry_value("y"),
                "width": _safe_entry_value("width"),
                "height": height_val,
            }

        colors = {key: rgb_to_hex(value) for key, value in self.color_configs.items()}

        return {
            "version": 1,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "resolution": list(self.current_video_resolution),
            "map_style": self.map_style_var.get(),
            "zoom_level": int(self.map_zoom_level_var.get()),
            "elements": element_configs,
            "colors": colors,
        }

    def _apply_resolution_from_preset(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        resolution = (int(width), int(height))
        label = next((lbl for lbl, res in self.resolution_presets if res == resolution), None)
        if label is None:
            label = f"Personnalisé ({resolution[0]}x{resolution[1]})"
            current_values = list(self.resolution_combo["values"])
            if label not in current_values:
                current_values.append(label)
                self.resolution_combo["values"] = current_values
            self._resolution_lookup[label] = resolution
        self.resolution_choice_var.set(label)
        self.set_video_resolution(resolution)

    def _apply_configuration_preset(self, config: dict) -> None:
        if not isinstance(config, dict):
            raise ValueError("format de preset invalide")

        resolution = config.get("resolution")
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            try:
                width = int(resolution[0])
                height = int(resolution[1])
            except (TypeError, ValueError):
                width = height = None
            if width and height:
                self._apply_resolution_from_preset(width, height)

        map_style = config.get("map_style")
        if isinstance(map_style, str) and map_style in MAP_TILE_SERVERS:
            self.map_style_var.set(map_style)
            self.map_style_combo.set(map_style)

        zoom_level = config.get("zoom_level")
        if isinstance(zoom_level, int) and 1 <= zoom_level <= 12:
            self.map_zoom_level_var.set(zoom_level)
            self.zoom_combo.set(str(zoom_level))

        colors = config.get("colors", {})
        if isinstance(colors, dict):
            for key, value in colors.items():
                hex_color = rgb_to_hex(value)
                self.color_configs[key] = hex_color
                if key in self.color_preview_frames:
                    self.color_preview_frames[key].config(background=hex_color)

        elements = config.get("elements", {})
        if isinstance(elements, dict):
            previous_block_state = self._block_recursion
            self._block_recursion = True
            try:
                for name, params in elements.items():
                    if name not in DEFAULT_ELEMENT_CONFIGS or not isinstance(params, dict):
                        continue
                    visible = params.get("visible")
                    if name in self.element_visibility_vars and visible is not None:
                        self.element_visibility_vars[name].set(bool(visible))
                    for key in ("x", "y", "width"):
                        if key not in params or name not in self.element_pos_entries_vars:
                            continue
                        try:
                            val = int(params[key])
                        except (TypeError, ValueError):
                            continue
                        max_val = (
                            self.current_video_resolution[0]
                            if key in ("x", "width")
                            else self.current_video_resolution[1]
                        )
                        val = max(0, min(val, max_val))
                        entry_var = self.element_pos_entries_vars[name][key]
                        slider_var = self.element_sliders_vars[name][key]
                        entry_var.set(str(val))
                        slider_var.set(val)
                    if name in self.element_calculated_height_labels and "height" in params:
                        try:
                            height_val = int(params["height"])
                        except (TypeError, ValueError):
                            height_val = DEFAULT_ELEMENT_CONFIGS[name]["height"]
                        self.element_calculated_height_labels[name].set(str(height_val))
            finally:
                self._block_recursion = previous_block_state

        self.show_preview(force_update=True)

    def save_configuration_preset(self) -> None:
        config = self._collect_current_configuration()
        initial_dir = os.path.dirname(self.gpx_file_path) if self.gpx_file_path else os.getcwd()
        filename = filedialog.asksaveasfilename(
            title="Enregistrer un preset",
            defaultextension=".json",
            filetypes=[("Preset GPX Overlay", "*.json"), ("Tous les fichiers", "*.*")],
            initialdir=initial_dir,
            initialfile="preset_overlay.json",
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as fh:
                json.dump(config, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            messagebox.showerror("Erreur", f"Impossible d'enregistrer le preset:\n{exc}")
            return
        messagebox.showinfo("Preset enregistré", f"Configuration enregistrée dans:\n{filename}")

    def load_configuration_preset(self) -> None:
        initial_dir = os.path.dirname(self.gpx_file_path) if self.gpx_file_path else os.getcwd()
        filename = filedialog.askopenfilename(
            title="Charger un preset",
            filetypes=[("Preset GPX Overlay", "*.json"), ("Tous les fichiers", "*.*")],
            initialdir=initial_dir,
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Erreur", f"Impossible de charger le preset:\n{exc}")
            return
        try:
            self._apply_configuration_preset(data)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Erreur", f"Preset invalide:\n{exc}")
            return
        messagebox.showinfo("Preset chargé", f"Preset chargé depuis:\n{filename}")

    def get_smoothing_seconds(self) -> float:
        text = self.graph_smoothing_seconds_var.get().strip()
        if not text:
            return DEFAULT_GRAPH_SMOOTHING_SECONDS
        try:
            value = float(text.replace(",", "."))
        except ValueError as exc:
            raise ValueError("invalid smoothing") from exc
        if value < 0:
            raise ValueError("invalid smoothing")
        return value

    # ----- Misc UI -----
    def on_resolution_selection_change(self, event=None):
        if self._block_recursion:
            return
        label = self.resolution_choice_var.get()
        new_resolution = self._resolution_lookup.get(label)
        if new_resolution:
            self.set_video_resolution(new_resolution)

    def set_video_resolution(self, new_resolution: tuple[int, int]):
        old_width, old_height = self.current_video_resolution
        new_width, new_height = new_resolution
        if old_width == new_width and old_height == new_height:
            return
        scale_x = new_width / old_width if old_width else 1.0
        scale_y = new_height / old_height if old_height else 1.0
        self._block_recursion = True
        try:
            self.current_video_resolution = [new_width, new_height]
            self.update_all_slider_ranges()
            self._scale_layout_for_resolution(scale_x, scale_y)
        finally:
            self._block_recursion = False
        self.show_preview(force_update=True)

    def _scale_layout_for_resolution(self, scale_x: float, scale_y: float) -> None:
        max_w, max_h = self.current_video_resolution
        for element_name in self.element_pos_entries_vars:
            for key in ("x", "y", "width"):
                entry_var = self.element_pos_entries_vars[element_name][key]
                try:
                    current_val = int(entry_var.get())
                except (ValueError, tk.TclError):
                    continue
                if key in ("x", "width"):
                    scaled = int(round(current_val * scale_x))
                    max_dim = max_w
                else:
                    scaled = int(round(current_val * scale_y))
                    max_dim = max_h
                scaled = max(0, min(scaled, max_dim))
                entry_var.set(str(scaled))
                self.element_sliders_vars[element_name][key].set(scaled)

            if element_name in self.element_initial_ratios:
                ratio = self.element_initial_ratios[element_name]
                try:
                    w_val = int(self.element_pos_entries_vars[element_name]["width"].get())
                except (ValueError, tk.TclError):
                    continue
                if ratio == float("inf"):
                    h_val = DEFAULT_ELEMENT_CONFIGS[element_name]["height"]
                else:
                    h_val = int(round(w_val * ratio))
                h_val = max(0, min(h_val, max_h))
                self.element_calculated_height_labels[element_name].set(str(h_val))

    def handle_element_change(self, element_name, key, source="slider"):
        if self._block_recursion:
            return
        if source == "slider":
            value = self.element_sliders_vars[element_name][key].get()
            self.element_pos_entries_vars[element_name][key].set(str(int(value)))
        else:
            text = self.element_pos_entries_vars[element_name][key].get()
            if text in ("", "-"): return
            try: value = int(text); self.element_sliders_vars[element_name][key].set(value)
            except ValueError: return

        if key in ["x", "width"]: max_val = self.current_video_resolution[0]
        else: max_val = self.current_video_resolution[1]
        val = int(self.element_pos_entries_vars[element_name][key].get()); val = 0 if val < 0 else min(val, max_val)
        if source == "slider": self.element_pos_entries_vars[element_name][key].set(str(val))
        else: self.element_sliders_vars[element_name][key].set(val)

        if key == "width" and element_name in self.element_initial_ratios:
            ratio = self.element_initial_ratios[element_name]
            try:
                w_val = int(self.element_pos_entries_vars[element_name]["width"].get())
                h_val = int(round(w_val * ratio)) if ratio != float("inf") else DEFAULT_ELEMENT_CONFIGS[element_name]["height"]
                self.element_calculated_height_labels[element_name].set(str(h_val))
            except Exception:
                pass
        self.show_preview(force_update=True)

    def update_all_slider_ranges(self):
        for element_name, sliders in self.element_sliders.items():
            for key, slider in sliders.items():
                if key in ["x", "width"]: slider.config(from_=0, to=self.current_video_resolution[0])
                else: slider.config(from_=0, to=self.current_video_resolution[1])

    def update_preview_area_size(self):
        self.preview_label.update_idletasks()
        self.preview_area_width = self.preview_label.winfo_width() or 640
        self.preview_area_height = self.preview_label.winfo_height() or 360

    def update_start_offset_max(self, *args):
        if not (self.gpx_start_time_raw and self.gpx_end_time_raw):
            return
        total_sec = int((self.gpx_end_time_raw - self.gpx_start_time_raw).total_seconds())
        try:
            dur = int(self.duration_var.get())
        except Exception:
            dur = 0
        post = int(self.post_roll_var.get()) if self.post_roll_var else 0
        max_start = max(0, total_sec - dur - post)
        self.start_offset_scale.config(to=max_start)
        if self.start_offset_var.get() > max_start:
            self.start_offset_var.set(max_start)

        self.update_start_offset_label()

    def update_duration_max(self, *args):
        if not (self.gpx_start_time_raw and self.gpx_end_time_raw):
            return
        total_sec = int((self.gpx_end_time_raw - self.gpx_start_time_raw).total_seconds())
        start = int(self.start_offset_var.get())
        post = int(self.post_roll_var.get()) if self.post_roll_var else 0
        max_dur = max(1, total_sec - start - post)
        self.duration_scale.config(to=max_dur)
        if self.duration_var.get() > max_dur:
            self.duration_var.set(max_dur)
        self.update_duration_label()

    def update_clip_time_label(self):
        if self.gpx_start_time_raw:
            start_dt = self.gpx_start_time_raw + timedelta(seconds=int(self.start_offset_var.get()))
            end_dt = start_dt + timedelta(seconds=int(self.duration_var.get()))
            self.clip_time_label.config(
                text=f"Clip: début {start_dt.strftime('%H:%M:%S')} - fin {end_dt.strftime('%H:%M:%S')}"
            )
        else:
            self.clip_time_label.config(text="Clip: début N/A - fin N/A")
        self.update_display_time_label()

    def update_display_time_label(self):
        if not hasattr(self, "display_time_label") or self.display_time_label is None:
            return
        if not self.gpx_start_time_raw or not self.gpx_end_time_raw:
            self.display_time_label.config(text="Fenêtre affichée: début N/A - fin N/A")
            return

        tz = pytz.timezone("Europe/Paris")
        clip_start = self.gpx_start_time_raw + timedelta(seconds=int(self.start_offset_var.get()))
        clip_end = clip_start + timedelta(seconds=int(self.duration_var.get()))
        display_start = clip_start - timedelta(seconds=int(self.pre_roll_var.get()))
        display_end = clip_end + timedelta(seconds=int(self.post_roll_var.get()))
        if display_start < self.gpx_start_time_raw:
            display_start = self.gpx_start_time_raw
        if display_end > self.gpx_end_time_raw:
            display_end = self.gpx_end_time_raw
        self.display_time_label.config(
            text=(
                f"Fenêtre affichée: début {display_start.astimezone(tz).strftime('%H:%M:%S')}"
                f" - fin {display_end.astimezone(tz).strftime('%H:%M:%S')}"
            )
        )

    def update_pre_roll_label(self, *args):
        if hasattr(self, "pre_roll_label"):
            self.pre_roll_label.config(text=f"{int(self.pre_roll_var.get())} s")

    def update_post_roll_label(self, *args):
        if hasattr(self, "post_roll_label"):
            self.post_roll_label.config(text=f"{int(self.post_roll_var.get())} s")

    def update_margin_scales(self):
        if not hasattr(self, "pre_roll_scale") or not hasattr(self, "post_roll_scale"):
            return
        if not (self.gpx_start_time_raw and self.gpx_end_time_raw):
            max_pre = int(self.start_offset_var.get())
            self.pre_roll_scale.config(to=max_pre)
            if self.pre_roll_var.get() > max_pre:
                self.pre_roll_var.set(max_pre)
            self.post_roll_scale.config(to=0)
            if self.post_roll_var.get() != 0:
                self.post_roll_var.set(0)
            return

        total_sec = int((self.gpx_end_time_raw - self.gpx_start_time_raw).total_seconds())
        start = int(self.start_offset_var.get())
        duration = int(self.duration_var.get())
        max_pre = max(0, start)
        self.pre_roll_scale.config(to=max_pre)
        if self.pre_roll_var.get() > max_pre:
            self.pre_roll_var.set(max_pre)
        max_post = max(0, total_sec - (start + duration))
        self.post_roll_scale.config(to=max_post)
        if self.post_roll_var.get() > max_post:
            self.post_roll_var.set(max_post)

    def update_start_offset_label(self, *args):
        start = int(self.start_offset_var.get())
        total = 0
        if self.gpx_start_time_raw and self.gpx_end_time_raw:
            total = int((self.gpx_end_time_raw - self.gpx_start_time_raw).total_seconds())
        remaining = max(0, total - start)
        self.start_offset_label.config(
            text=f"{format_hms(start)} (reste {format_hms(remaining)})"
        )

    def update_duration_label(self, *args):
        dur = int(self.duration_var.get())
        self.duration_label.config(text=format_hms(dur))

    def on_start_offset_change(self, *args):
        self.update_start_offset_label()
        self.update_duration_max()
        self.update_clip_time_label()
        self.update_margin_scales()

    def on_duration_slider_change(self, *args):
        self.update_duration_label()
        self.update_start_offset_max()
        self.update_clip_time_label()
        self.update_margin_scales()

    def on_pre_roll_change(self, *args):
        self.update_pre_roll_label()
        self.update_margin_scales()
        self.update_display_time_label()

    def on_post_roll_change(self, *args):
        self.update_post_roll_label()
        self.update_start_offset_max()
        self.update_duration_max()
        self.update_margin_scales()
        self.update_display_time_label()

    def show_preview(self, force_update: bool = False, initial_load: bool = False) -> None:
        if not initial_load and not force_update: return
        element_configs = {}
        for name in DEFAULT_ELEMENT_CONFIGS.keys():
            h_str = self.element_calculated_height_labels.get(name, tk.StringVar()).get()
            h = int(h_str) if h_str else DEFAULT_ELEMENT_CONFIGS[name]["height"]
            element_configs[name] = {
                "visible": self.element_visibility_vars.get(name, tk.BooleanVar(value=DEFAULT_ELEMENT_CONFIGS[name]["visible"])).get(),
                "x": int(self.element_pos_entries_vars.get(name, {}).get("x", tk.StringVar(value=DEFAULT_ELEMENT_CONFIGS[name]["x"])).get()),
                "y": int(self.element_pos_entries_vars.get(name, {}).get("y", tk.StringVar(value=DEFAULT_ELEMENT_CONFIGS[name]["y"])).get()),
                "width": int(self.element_pos_entries_vars.get(name, {}).get("width", tk.StringVar(value=DEFAULT_ELEMENT_CONFIGS[name]["width"])).get()),
                "height": h,
            }
        preview_img = generate_preview_image(
            tuple(self.current_video_resolution),
            self.font_path_var.get() or DEFAULT_FONT_PATH,
            element_configs,
            self.color_configs,
        )
        try:
            target_w = max(1, self.preview_area_width); target_h = max(1, self.preview_area_height)
            if preview_img.width > target_w or preview_img.height > target_h:
                ratio = min(target_w / preview_img.width, target_h / preview_img.height)
                preview_img = preview_img.resize((int(preview_img.width * ratio), int(preview_img.height * ratio)), Image.LANCZOS)
        except Exception:
            pass
        photo = ImageTk.PhotoImage(preview_img); self.preview_image_tk = photo
        self.preview_label.configure(image=photo); self.preview_label.image = photo


    def preview_first_frame(self):
        if not self.gpx_file_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier GPX.")
            return
        try:
            duration = max(1, int(self.duration_var.get()))
            fps = max(1, int(self.fps_entry_var.get()))
        except Exception:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides (durée, FPS).")
            return

        try:
            smoothing_seconds = self.get_smoothing_seconds()
        except ValueError:
            messagebox.showerror(
                "Erreur",
                "Veuillez saisir une durée de lissage valide (secondes, valeur ≥ 0).",
            )
            return

        element_configs = {}
        for name in DEFAULT_ELEMENT_CONFIGS.keys():
            h_str = self.element_calculated_height_labels.get(name, tk.StringVar()).get()
            h = int(h_str) if h_str else DEFAULT_ELEMENT_CONFIGS[name]["height"]
            element_configs[name] = {
                "visible": self.element_visibility_vars.get(name, tk.BooleanVar(value=DEFAULT_ELEMENT_CONFIGS[name]["visible"])).get(),
                "x": int(self.element_pos_entries_vars.get(name, {}).get("x", tk.StringVar(value=DEFAULT_ELEMENT_CONFIGS[name]["x"])).get()),
                "y": int(self.element_pos_entries_vars.get(name, {}).get("y", tk.StringVar(value=DEFAULT_ELEMENT_CONFIGS[name]["y"])).get()),
                "width": int(self.element_pos_entries_vars.get(name, {}).get("width", tk.StringVar(value=DEFAULT_ELEMENT_CONFIGS[name]["width"])).get()),
                "height": h,
            }
        res = tuple(self.current_video_resolution)
        font_path = self.font_path_var.get() or DEFAULT_FONT_PATH
        font_size = max(10, int(self.font_size_var.get()))
        global FONT_SIZE_LARGE, FONT_SIZE_MEDIUM
        FONT_SIZE_LARGE = font_size
        FONT_SIZE_MEDIUM = int(font_size * 0.75)

        try:
            img = render_first_frame_image(
                gpx_filename=self.gpx_file_path,
                start_offset=int(self.start_offset_var.get()),
                clip_duration=duration,
                fps=fps,
                resolution=res,
                font_path=font_path,
                element_configs=element_configs,
                color_configs=self.color_configs,
                map_style=self.map_style_var.get(),
                zoom_level_ui=int(self.map_zoom_level_var.get()),
                smoothing_seconds=smoothing_seconds,
                pre_roll_seconds=int(self.pre_roll_var.get()),
                post_roll_seconds=int(self.post_roll_var.get()),
            )
        except Exception as e:
            messagebox.showerror("Erreur", f"Aperçu impossible: {e}")
            return

        self.update_preview_area_size()
        target_w = max(1, self.preview_area_width)
        target_h = max(1, self.preview_area_height)
        try:
            if img.width > target_w or img.height > target_h:
                r = min(target_w / img.width, target_h / img.height)
                img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))), Image.LANCZOS)
        except Exception:
            pass
        photo = ImageTk.PhotoImage(img)
        self.preview_image_tk = photo
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def select_font_file(self):
        filetypes = [("Fichiers de police", "*.ttf *.otf"), ("Tous les fichiers", "*.*")]
        filename = filedialog.askopenfilename(title="Choisir une police", filetypes=filetypes)
        if filename:
            self.font_path_var.set(filename)

    def select_gpx_file(self):
        filetypes = [("Fichiers GPX", "*.gpx"), ("Tous les fichiers", "*.*")]
        filename = filedialog.askopenfilename(title="Choisir un fichier GPX", filetypes=filetypes)
        if filename:
            self.gpx_file_path = filename
            points, start, end = parse_gpx(filename)
            if not points:
                messagebox.showerror("Erreur", "Impossible de charger le fichier GPX."); return
            self.gpx_start_time_raw = start; self.gpx_end_time_raw = end
            base = os.path.basename(filename)
            self.start_offset_var.set(0)
            self.pre_roll_var.set(0)
            self.post_roll_var.set(0)

            self.update_duration_max()

            self.update_start_offset_max()
            self.update_margin_scales()
        self.gpx_label.config(text=f"Fichier GPX: {base}"); self.gpx_toolbar_label_var.set(f"GPX: {base}")
        tz = pytz.timezone('Europe/Paris')
        self.gpx_start_time_label.config(text=f"Début GPX: {start.astimezone(tz).strftime('%Y-%m-%d %H:%M:%S')}" if start else "Début GPX: N/A")
        self.gpx_end_time_label.config(text=f"Fin GPX: {end.astimezone(tz).strftime('%Y-%m-%d %H:%M:%S')}" if end else "Fin GPX: N/A")
        if start and end:
            total_sec = int((end - start).total_seconds())
            from datetime import timedelta
            self.gpx_duration_label.config(text=f"Durée GPX: {str(timedelta(seconds=total_sec))}")
        else:
            self.gpx_duration_label.config(text="Durée GPX: N/A")

        self.update_duration_label()
        self.update_pre_roll_label()
        self.update_post_roll_label()
        self.update_display_time_label()

    def on_preview_resize(self, event) -> None:
        self.update_preview_area_size()
        if self.preview_image_tk: self.show_preview(force_update=True)

    def generate_video(self):
        if not self.gpx_file_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier GPX."); return
        try:
            duration = int(self.duration_var.get()); fps = int(self.fps_entry_var.get())
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides."); return

        try:
            smoothing_seconds = self.get_smoothing_seconds()
        except ValueError:
            messagebox.showerror(
                "Erreur",
                "Veuillez saisir une durée de lissage valide (secondes, valeur ≥ 0).",
            )
            return

        element_configs = {}
        for name in DEFAULT_ELEMENT_CONFIGS.keys():
            element_configs[name] = {
                "visible": self.element_visibility_vars[name].get(),
                "x": int(self.element_pos_entries_vars[name]["x"].get()),
                "y": int(self.element_pos_entries_vars[name]["y"].get()),
                "width": int(self.element_pos_entries_vars[name]["width"].get()),
                "height": int(self.element_calculated_height_labels[name].get()) if self.element_calculated_height_labels[name].get() else DEFAULT_ELEMENT_CONFIGS[name]["height"],
            }

        resolution = tuple(self.current_video_resolution)
        output_dir = os.path.dirname(self.gpx_file_path) or os.getcwd()
        base_name = os.path.splitext(os.path.basename(self.gpx_file_path))[0] or "video"
        output_file = os.path.join(output_dir, f"{base_name}.mp4")
        if os.path.exists(output_file):
            overwrite = messagebox.askyesno(
                "Fichier existant",
                f"Le fichier \"{output_file}\" existe déjà. Voulez-vous l'écraser ?",
            )
            if not overwrite:
                return

        self.generate_btn.config(state=tk.DISABLED)
        self.progress_time_var.set("Temps restant estimé : calcul en cours…")
        start_time = time.time()

        def run_generation():
            error_msg = None

            def progress_cb(pct):
                if pct > 0:
                    elapsed = time.time() - start_time
                    remaining = elapsed * (100 - pct) / pct
                    label = f"Temps restant estimé : {format_hms(int(remaining))}"
                else:
                    label = "Temps restant estimé : calcul en cours…"

                def updater():
                    self.progress_time_var.set(label)
                self.master.after(0, updater)
        
            try:
                font_path = self.font_path_var.get() or DEFAULT_FONT_PATH
                font_size = max(10, int(self.font_size_var.get()))
                global FONT_SIZE_LARGE, FONT_SIZE_MEDIUM
                FONT_SIZE_LARGE = font_size
                FONT_SIZE_MEDIUM = int(font_size * 0.75)
        
                success = generate_gpx_video(
                    self.gpx_file_path,
                    output_file,
                    int(self.start_offset_var.get()),
                    duration,
                    fps,
                    resolution,
                    font_path,
                    element_configs,
                    self.color_configs,
                    map_style=self.map_style_var.get(),
                    zoom_level_ui=int(self.map_zoom_level_var.get()),
                    smoothing_seconds=smoothing_seconds,
                    progress_callback=progress_cb,
                    pre_roll_seconds=int(self.pre_roll_var.get()),
                    post_roll_seconds=int(self.post_roll_var.get()),
                )
            except Exception as e:
                success = False
                error_msg = str(e)

            def finalize():
                self.generate_btn.config(state=tk.NORMAL)
                self.progress_time_var.set(self.progress_message_default)
                if success:
                    messagebox.showinfo(
                        "Succès",
                        "La vidéo a été générée avec succès dans :\n"
                        f"{output_file}",
                    )
                else:
                    if error_msg:
                        messagebox.showerror("Erreur", f"Échec génération : {error_msg}")
                    else:
                        messagebox.showerror("Erreur", "Une erreur est survenue lors de la génération de la vidéo.")
        
            self.master.after(0, finalize)


        threading.Thread(target=run_generation, daemon=True).start()


# ----- Main -----
if __name__ == "__main__":
    root = tk.Tk()
    app = GPXVideoApp(root)
    root.mainloop()