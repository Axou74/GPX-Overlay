# Overlay_dynamique.py
# GPX -> Vidéo avec carte fond (StaticMap), zoom 1..12 avec centre/zoom explicites,
# emprise large fiable (WebMercator), durée par défaut 20s.

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys, time
import threading
import imageio.v2 as imageio

import numpy as np
import pytz
from PIL import Image, ImageDraw, ImageFont, ImageTk

from rendering import TrackData, load_interpolated_track, parse_gpx

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
PATH_COLOR = (200, 200, 200)
CURRENT_PATH_COLOR = (255, 255, 255)
CURRENT_POINT_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
GAUGE_BG_COLOR = (30, 30, 30)

FONT_SIZE_LARGE = 40
FONT_SIZE_MEDIUM = 20
GRAPH_FONT_SCALE = 0.7
MIN_GRAPH_FONT_SIZE = 4
MARGIN = 10
GRAPH_PADDING = 100

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


@dataclass
class RenderContext:
    map_area: dict
    map_size: tuple[int, int]
    map_center: tuple[float, float]
    graph_layers: list[dict]
    gauge_circ_area: dict
    gauge_lin_area: dict
    gauge_cnt_area: dict
    compass_area: dict
    info_area: dict
    speed_bounds: tuple[float, float]


def prepare_render_style(font_path: str, color_configs: dict | None):
    graph_font_size = compute_graph_font_size(FONT_SIZE_MEDIUM)
    try:
        font_large = ImageFont.truetype(font_path, FONT_SIZE_LARGE)
        font_medium = ImageFont.truetype(font_path, FONT_SIZE_MEDIUM)
        font_graph = ImageFont.truetype(font_path, graph_font_size)
    except IOError:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_graph = ImageFont.load_default()

    colors = {
        "background": color_configs.get("background", BG_COLOR) if color_configs else BG_COLOR,
        "map_path": color_configs.get("map_path", PATH_COLOR) if color_configs else PATH_COLOR,
        "map_current_path": color_configs.get("map_current_path", CURRENT_PATH_COLOR) if color_configs else CURRENT_PATH_COLOR,
        "map_current_point": color_configs.get("map_current_point", CURRENT_POINT_COLOR) if color_configs else CURRENT_POINT_COLOR,
        "graph_altitude": color_configs.get("graph_altitude", PATH_COLOR) if color_configs else PATH_COLOR,
        "graph_speed": color_configs.get("graph_speed", PATH_COLOR) if color_configs else PATH_COLOR,
        "graph_pace": color_configs.get("graph_pace", PATH_COLOR) if color_configs else PATH_COLOR,
        "graph_hr": color_configs.get("graph_hr", PATH_COLOR) if color_configs else PATH_COLOR,
        "graph_current_point": color_configs.get("graph_current_point", CURRENT_POINT_COLOR) if color_configs else CURRENT_POINT_COLOR,
        "text": color_configs.get("text", TEXT_COLOR) if color_configs else TEXT_COLOR,
        "gauge_background": color_configs.get("gauge_background", GAUGE_BG_COLOR) if color_configs else GAUGE_BG_COLOR,
    }

    fonts = {
        "large": font_large,
        "medium": font_medium,
        "graph": font_graph,
    }

    return fonts, colors


def _safe_area(area: dict | None) -> dict:
    if area:
        return area
    return {"x": 0, "y": 0, "width": 1, "height": 1, "visible": False}


def prepare_render_context(
    resolution: tuple[int, int],
    element_configs: dict,
    track: TrackData,
    font_graph,
    colors: dict,
) -> RenderContext:
    map_area = element_configs.get("Carte", {})
    if not is_area_visible(map_area):
        raise ValueError("Zone carte non visible ou dimensions nulles.")
    mw = int(map_area.get("width", 0))
    mh = int(map_area.get("height", 0))

    elev_area = element_configs.get("Profil Altitude", {})
    speed_area = element_configs.get("Profil Vitesse", {})
    pace_area = element_configs.get("Profil Allure", {})
    hr_area = element_configs.get("Profil Cardio", {})
    gauge_circ_area = element_configs.get("Jauge Vitesse Circulaire", {})
    gauge_lin_area = element_configs.get("Jauge Vitesse Linéaire", {})
    gauge_cnt_area = element_configs.get("Compteur de vitesse", {})
    compass_area = element_configs.get("Boussole (ruban)", {})
    info_area = element_configs.get("Infos Texte", {})

    elev_min = float(np.min(track.interp_eles))
    elev_max = float(np.max(track.interp_eles))
    elev_tf = GraphTransformer(elev_min, elev_max, _safe_area(elev_area))
    elev_path = [elev_tf.to_xy(i, val, len(track.interp_eles)) for i, val in enumerate(track.interp_eles)]

    speed_min_val = float(np.min(track.interp_speeds))
    speed_max_val = float(np.max(track.interp_speeds))
    speed_min, speed_max = auto_speed_bounds(track.interp_speeds)
    speed_tf = GraphTransformer(speed_min, speed_max, _safe_area(speed_area))
    speed_path = [speed_tf.to_xy(i, val, len(track.interp_speeds)) for i, val in enumerate(track.interp_speeds)]

    pace_min, pace_max = 0.0, 20.0
    pace_tf = GraphTransformer(pace_min, pace_max, _safe_area(pace_area))
    pace_path = [
        pace_tf.to_xy(i, (min(val, pace_max) if np.isfinite(val) else pace_max), len(track.interp_pace))
        for i, val in enumerate(track.interp_pace)
    ]

    has_hr = np.isfinite(track.interp_hrs).sum() >= 1
    if has_hr:
        hr_vals = track.interp_hrs[np.isfinite(track.interp_hrs)]
        hr_min, hr_max = float(np.min(hr_vals)), float(np.max(hr_vals))
    else:
        hr_min, hr_max = 0.0, 1.0
    hr_max = hr_max if hr_max > hr_min else (hr_min + 1.0)
    hr_tf = GraphTransformer(hr_min, hr_max, _safe_area(hr_area))
    hr_path = [
        hr_tf.to_xy(i, (val if np.isfinite(val) else hr_min), len(track.interp_hrs))
        for i, val in enumerate(track.interp_hrs)
    ]

    graph_specs = [
        (elev_area, elev_path, elev_min, elev_max, "Altitude", "m", colors["graph_altitude"]),
        (speed_area, speed_path, speed_min_val, speed_max_val, "Vitesse", "km/h", colors["graph_speed"]),
        (pace_area, pace_path, pace_min, pace_max, "Allure", "min/km", colors["graph_pace"]),
    ]
    if has_hr:
        graph_specs.append((hr_area, hr_path, hr_min, hr_max, "FC", "bpm", colors["graph_hr"]))

    graph_layers = prepare_graph_layers(
        resolution,
        font_graph,
        colors["text"],
        colors["graph_current_point"],
        graph_specs,
    )

    lon_c = (track.lon_min_raw + track.lon_max_raw) * 0.5
    lat_c = (track.lat_min_raw + track.lat_max_raw) * 0.5

    return RenderContext(
        map_area=map_area,
        map_size=(mw, mh),
        map_center=(lon_c, lat_c),
        graph_layers=graph_layers,
        gauge_circ_area=gauge_circ_area,
        gauge_lin_area=gauge_lin_area,
        gauge_cnt_area=gauge_cnt_area,
        compass_area=compass_area,
        info_area=info_area,
        speed_bounds=(speed_min, speed_max),
    )

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
    zoom_level_ui: int = 8,
    progress_callback=None,
) -> bool:
    PATCH_FACTOR = 2.4
    MAX_LARGE_DIM = 4096
    VERTICAL_BIAS = 0.65

    fonts, colors = prepare_render_style(font_path, color_configs or {})
    font_medium = fonts["medium"]
    font_graph = fonts["graph"]

    try:
        track = load_interpolated_track(
            gpx_filename,
            start_offset,
            clip_duration,
            fps,
        )
    except ValueError as e:
        print(f"Erreur: {e}")
        return False

    try:
        ui_context = prepare_render_context(
            resolution,
            element_configs,
            track,
            font_graph,
            colors,
        )
    except ValueError as e:
        print(f"Erreur: {e}")
        return False

    total_frames = track.total_frames
    print(f"Génération de {total_frames} images…")
    t0 = time.time()
    report_every = max(1, int(fps))

    def _progress(done: int):
        pct = int(done * 100 / total_frames) if total_frames else 100
        if progress_callback:
            progress_callback(pct)
        elif done == total_frames or (done % report_every == 0):
            elapsed = time.time() - t0
            fps_eff = (done / elapsed) if elapsed > 0 else 0.0
            eta = (total_frames - done) / fps_eff if fps_eff > 0 else 0.0
            sys.stdout.write(
                f"\rFrames {done}/{total_frames}  | {pct}%  | {fps_eff:.1f} fps  | ETA {eta:0.0f}s"
            )
            sys.stdout.flush()

    interp_times = track.interp_times
    interp_lats = track.interp_lats
    interp_lons = track.interp_lons
    interp_eles = track.interp_eles
    interp_speeds = track.interp_speeds
    interp_slopes = track.interp_slopes
    interp_pace = track.interp_pace
    interp_hrs = track.interp_hrs

    map_area = ui_context.map_area
    mw, mh = ui_context.map_size
    graph_layers = ui_context.graph_layers
    gauge_circ_area = ui_context.gauge_circ_area
    gauge_lin_area = ui_context.gauge_lin_area
    gauge_cnt_area = ui_context.gauge_cnt_area
    compass_area = ui_context.compass_area
    info_area = ui_context.info_area
    speed_min, speed_max = ui_context.speed_bounds

    lon_c, lat_c = ui_context.map_center
    lon_min_raw = track.lon_min_raw
    lon_max_raw = track.lon_max_raw
    lat_min_raw = track.lat_min_raw
    lat_max_raw = track.lat_max_raw
    start_time = track.start_time
    tz = track.timezone

    bg_c = colors["background"]
    map_path_c = colors["map_path"]
    map_current_path_c = colors["map_current_path"]
    map_current_point_c = colors["map_current_point"]
    gauge_bg_c = colors["gauge_background"]
    text_c = colors["text"]

    try:
        writer = imageio.get_writer(output_filename, fps=fps, codec="libx264", macro_block_size=1)

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
                width_large,
                height_large,
                map_style,
                zoom,
                lon_c,
                lat_c,
                bg_c,
                fail_on_tile_error=True,
            )
        except Exception as e:
            print(f"Fond carte non dispo (dyn), fond uni utilisé: {e}")
            base_map_img_large = Image.new("RGB", (width_large, height_large), bg_c)

        x_full = xs_world - x0_world
        y_full = ys_world - y0_world
        global_xy = np.column_stack((np.rint(x_full).astype(int), np.rint(y_full).astype(int)))
        local_xy_buffer = np.empty_like(global_xy)

        if total_frames == 0:
            smoothed_angles = np.zeros(0, dtype=float)
        else:
            dx = np.zeros(total_frames, dtype=float)
            dy = np.zeros(total_frames, dtype=float)
            if total_frames > 1:
                dx[:-1] = np.diff(x_full)
                dy[:-1] = np.diff(y_full)
                dx[-1] = x_full[-1] - x_full[-2]
                dy[-1] = y_full[-1] - y_full[-2]
            headings = np.zeros(total_frames, dtype=float)
            non_zero = (np.abs(dx) > 1e-9) | (np.abs(dy) > 1e-9)
            headings[non_zero] = np.arctan2(dx[non_zero], -dy[non_zero])

            complex_raw = np.exp(1j * headings)
            win_sizes = np.clip((15 - interp_speeds).astype(int), 3, 15)
            half_windows = win_sizes // 2
            indices = np.arange(total_frames)
            start_idx = np.maximum(indices - half_windows, 0)
            end_idx = np.minimum(indices + half_windows + 1, total_frames)

            cumulative = np.concatenate(([0.0 + 0.0j], np.cumsum(complex_raw)))
            counts = (end_idx - start_idx).astype(float)
            counts[counts == 0] = 1.0
            averages = (cumulative[end_idx] - cumulative[start_idx]) / counts
            smoothed_angles = np.arctan2(averages.imag, averages.real)

        last_heading_deg = math.degrees(smoothed_angles[0]) if total_frames else 0.0

        for frame_idx in range(total_frames):
            frame_img = Image.new("RGB", resolution, bg_c)
            draw = ImageDraw.Draw(frame_img)

            xc = float(x_full[frame_idx]); yc = float(y_full[frame_idx])
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
            pdraw.line(local_xy_list[: frame_idx + 1], fill=map_current_path_c, width=4)
            cxp, cyp = local_xy_buffer[frame_idx]
            r = 6
            pdraw.ellipse((int(cxp - r), int(cyp - r), int(cxp + r), int(cyp + r)), fill=map_current_point_c)

            speed_kmh = float(interp_speeds[frame_idx])
            desired_heading = math.degrees(smoothed_angles[frame_idx])
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

            view_left = int(round(patch_w / 2.0 - mw / 2.0))
            view_top = int(round(patch_h / 2.0 - VERTICAL_BIAS * mh))
            view = patch_img.crop((view_left, view_top, view_left + mw, view_top + mh))

            frame_img.paste(view, (int(map_area.get("x", 0)), int(map_area.get("y", 0))))
            draw_north_arrow(frame_img, map_area, heading_deg, text_c)

            for layer in graph_layers:
                frame_img.paste(layer["background"], (0, 0), layer["background"])
            for layer in graph_layers:
                draw_graph_progress_overlay(
                    draw,
                    layer["path"],
                    frame_idx,
                    layer["base_color"],
                    layer["point_color"],
                    layer["point_size"],
                )

            if gauge_circ_area.get("visible", False):
                draw_circular_speedometer(
                    draw,
                    float(interp_speeds[frame_idx]),
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
                    float(interp_speeds[frame_idx]),
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
                    float(interp_speeds[frame_idx]),
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
                draw_info_text(
                    draw,
                    float(interp_speeds[frame_idx]),
                    float(interp_eles[frame_idx]),
                    float(interp_slopes[frame_idx]),
                    start_time + timedelta(seconds=float(interp_times[frame_idx])),
                    info_area,
                    font_medium,
                    tz,
                    text_c,
                )
                pace_now = pace_min_per_km_from_speed_kmh(float(interp_speeds[frame_idx]))
                hr_now = float(interp_hrs[frame_idx]) if np.isfinite(interp_hrs[frame_idx]) else None
                draw_pace_hr_text(draw, pace_now, hr_now, info_area, font_medium, text_c)

            writer.append_data(np.array(frame_img))
            _progress(frame_idx + 1)

        writer.close()
        print("Vidéo générée avec succès!")
        return True

    except Exception as e:
        print(f"Erreur écriture vidéo: {e}")
        return False


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
):
    PATCH_FACTOR = 2.4
    MAX_LARGE_DIM = 4096
    VERTICAL_BIAS = 0.65

    fonts, colors = prepare_render_style(font_path, color_configs or {})
    font_medium = fonts["medium"]
    font_graph = fonts["graph"]

    track = load_interpolated_track(
        gpx_filename,
        start_offset,
        clip_duration,
        fps,
    )

    ui_context = prepare_render_context(
        resolution,
        element_configs,
        track,
        font_graph,
        colors,
    )

    map_area = ui_context.map_area
    mw, mh = ui_context.map_size
    graph_layers = ui_context.graph_layers
    gauge_circ_area = ui_context.gauge_circ_area
    gauge_lin_area = ui_context.gauge_lin_area
    gauge_cnt_area = ui_context.gauge_cnt_area
    compass_area = ui_context.compass_area
    info_area = ui_context.info_area
    speed_min, speed_max = ui_context.speed_bounds

    lon_c, lat_c = ui_context.map_center
    lon_min_raw = track.lon_min_raw
    lon_max_raw = track.lon_max_raw
    lat_min_raw = track.lat_min_raw
    lat_max_raw = track.lat_max_raw
    start_time = track.start_time
    tz = track.timezone

    bg_c = colors["background"]
    map_path_c = colors["map_path"]
    map_current_path_c = colors["map_current_path"]
    map_current_point_c = colors["map_current_point"]
    gauge_bg_c = colors["gauge_background"]
    text_c = colors["text"]

    frame_img = Image.new("RGB", resolution, bg_c)
    draw = ImageDraw.Draw(frame_img)

    est_w = int(min(MAX_LARGE_DIM, mw * 6))
    est_h = int(min(MAX_LARGE_DIM, mh * 6))
    base_zoom = bbox_fit_zoom(est_w, est_h, lon_min_raw, lat_min_raw, lon_max_raw, lat_max_raw, padding_px=20)
    zoom = max(1, min(19, base_zoom + (zoom_level_ui - 8)))

    xs_world, ys_world = lonlat_to_pixel_np(track.interp_lons, track.interp_lats, zoom)
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
            width_large,
            height_large,
            map_style,
            zoom,
            lon_c,
            lat_c,
            bg_c,
            fail_on_tile_error=True,
        )
    except Exception as e:
        print(f"Fond carte non dispo (aperçu), fond uni utilisé: {e}")
        base_map_img_large = Image.new("RGB", (width_large, height_large), bg_c)

    x_full = xs_world - x0_world
    y_full = ys_world - y0_world
    global_xy = np.column_stack((np.rint(x_full).astype(int), np.rint(y_full).astype(int)))
    local_xy_buffer = np.empty_like(global_xy)

    total_frames = track.total_frames
    if total_frames == 0:
        smoothed_angles = np.zeros(0, dtype=float)
    else:
        dx = np.zeros(total_frames, dtype=float)
        dy = np.zeros(total_frames, dtype=float)
        if total_frames > 1:
            dx[:-1] = np.diff(x_full)
            dy[:-1] = np.diff(y_full)
            dx[-1] = x_full[-1] - x_full[-2]
            dy[-1] = y_full[-1] - y_full[-2]
        headings = np.zeros(total_frames, dtype=float)
        non_zero = (np.abs(dx) > 1e-9) | (np.abs(dy) > 1e-9)
        headings[non_zero] = np.arctan2(dx[non_zero], -dy[non_zero])
        complex_raw = np.exp(1j * headings)
        win_sizes = np.clip((15 - track.interp_speeds).astype(int), 3, 15)
        half_windows = win_sizes // 2
        indices = np.arange(total_frames)
        start_idx = np.maximum(indices - half_windows, 0)
        end_idx = np.minimum(indices + half_windows + 1, total_frames)
        cumulative = np.concatenate(([0.0 + 0.0j], np.cumsum(complex_raw)))
        counts = (end_idx - start_idx).astype(float)
        counts[counts == 0] = 1.0
        averages = (cumulative[end_idx] - cumulative[start_idx]) / counts
        smoothed_angles = np.arctan2(averages.imag, averages.real)

    frame_idx = 0
    xc = float(x_full[frame_idx]); yc = float(y_full[frame_idx])
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
    pdraw.line(local_xy_list[: frame_idx + 1], fill=map_current_path_c, width=4)
    cxp, cyp = local_xy_buffer[frame_idx]
    r = 6
    pdraw.ellipse((int(cxp - r), int(cyp - r), int(cxp + r), int(cyp + r)), fill=map_current_point_c)

    speed_kmh0 = float(track.interp_speeds[frame_idx])
    heading_deg = 0.0 if speed_kmh0 < 4.0 else math.degrees(smoothed_angles[frame_idx])
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

    for layer in graph_layers:
        frame_img.paste(layer["background"], (0, 0), layer["background"])
    for layer in graph_layers:
        draw_graph_progress_overlay(
            draw,
            layer["path"],
            frame_idx,
            layer["base_color"],
            layer["point_color"],
            layer["point_size"],
        )

    if gauge_circ_area.get("visible", False):
        draw_circular_speedometer(
            draw,
            speed_kmh0,
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
            speed_kmh0,
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
            speed_kmh0,
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
        draw_info_text(
            draw,
            speed_kmh0,
            float(track.interp_eles[frame_idx]),
            float(track.interp_slopes[frame_idx]),
            start_time + timedelta(seconds=float(track.interp_times[frame_idx])),
            info_area,
            font_medium,
            tz,
            text_c,
        )
        pace_now = pace_min_per_km_from_speed_kmh(speed_kmh0)
        hr_now = float(track.interp_hrs[frame_idx]) if np.isfinite(track.interp_hrs[frame_idx]) else None
        draw_pace_hr_text(draw, pace_now, hr_now, info_area, font_medium, text_c)

    return frame_img




# ----- Main -----
if __name__ == "__main__":
    root = tk.Tk()
    app = GPXVideoApp(root)
    root.mainloop()