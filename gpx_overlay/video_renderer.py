"""Rendering utilities for GPX overlays."""
from __future__ import annotations

import math
from datetime import timedelta

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import (
    BG_COLOR,
    FONT_SIZE_LARGE,
    FONT_SIZE_MEDIUM,
    GAUGE_BG_COLOR,
    MAP_TILE_SERVERS,
    PATH_COLOR,
    CURRENT_PATH_COLOR,
    CURRENT_POINT_COLOR,
    TEXT_COLOR,
)
from .gpx_parser import (
    format_pace_mmss,
    parse_gpx,
    filter_points_by_time,
    prepare_track_arrays,
)
from .map_tiles import bbox_fit_zoom, zoom_level_ui_to_offset, make_static_map

try:  # Optional dependency
    from staticmap import Line, CircleMarker  # type: ignore
    STATICMAP_AVAILABLE = True
except Exception:  # pragma: no cover - optional lib
    STATICMAP_AVAILABLE = False


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


def auto_speed_bounds(speeds: np.ndarray) -> tuple[float, float]:
    """Determine appropriate speed axis bounds based on observed speeds."""
    max_speed = float(np.max(speeds)) if speeds.size else 0.0
    if max_speed < 25:
        return 0.0, 25.0
    if max_speed < 80:
        return 0.0, 80.0
    return 0.0, float(math.ceil(max_speed / 20.0) * 20.0)


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
    draw.text((draw_area["x"], draw_area["y"] + draw_area["height"] + 10), f"Min {title}: {min_val:.0f} {unit}", font=font, fill=text_color)
    draw.text((draw_area["x"] + draw_area["width"] - 200, draw_area["y"] + draw_area["height"] + 10), f"Max {title}: {max_val:.0f} {unit}", font=font, fill=text_color)


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
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    draw.text((cx - text_w/2, y0 + (h - text_h)/2 - 10), speed_text, font=font, fill=text_color)


def draw_info_text(draw, speed, altitude, slope, current_time, draw_area, font, tz, text_color):
    display_time = current_time.astimezone(tz).strftime("%H:%M:%S")
    draw.text((draw_area["x"], draw_area["y"]), f"Vitesse : {speed:.0f} km/h", font=font, fill=text_color)
    draw.text((draw_area["x"], draw_area["y"] + FONT_SIZE_LARGE + 10), f"Altitude : {altitude:.0f} m", font=font, fill=text_color)
    draw.text((draw_area["x"], draw_area["y"] + 2 * (FONT_SIZE_LARGE + 10)), f"Heure : {display_time}", font=font, fill=text_color)
    draw.text((draw_area["x"], draw_area["y"] + 3 * (FONT_SIZE_LARGE + 10)), f"Pente : {slope:.1f} %", font=font, fill=text_color)


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
    fixed_map_view: bool = False,
    progress_callback=None,
) -> bool:
    """Generate a full GPX overlay video following the original script features."""

    points, gpx_start, _ = parse_gpx(gpx_filename)
    if not points or gpx_start is None:
        return False

    start_time = gpx_start + timedelta(seconds=float(start_offset))
    try:
        filtered, start_dt, tz = filter_points_by_time(
            points, start_time.isoformat(), clip_duration, "Europe/Paris"
        )
    except ValueError:
        return False

    times_seconds = np.array(
        [(p["time"].astimezone(tz) - start_dt).total_seconds() for p in filtered],
        dtype=float,
    )
    lats = np.array([p["lat"] for p in filtered], dtype=float)
    lons = np.array([p["lon"] for p in filtered], dtype=float)
    eles = np.array([p["ele"] for p in filtered], dtype=float)
    hrs_raw = np.array([
        p["hr"] if p.get("hr") is not None else np.nan for p in filtered
    ])

    track = prepare_track_arrays(
        times_seconds, lats, lons, eles, hrs_raw, clip_duration, fps
    )
    total_frames = track["total_frames"]

    coords = list(zip(track["interp_lons"], track["interp_lats"]))
    lon_min, lon_max = float(np.min(track["interp_lons"])), float(
        np.max(track["interp_lons"])
    )
    lat_min, lat_max = float(np.min(track["interp_lats"])), float(
        np.max(track["interp_lats"])
    )

    map_cfg = element_configs.get("Carte", {})
    map_width = map_cfg.get("width", resolution[0] // 2)
    map_height = map_cfg.get("height", resolution[1] // 2)

    zoom_base = bbox_fit_zoom(
        map_width, map_height, lon_min, lat_min, lon_max, lat_max, padding_px=20
    )
    zoom = zoom_base + zoom_level_ui_to_offset(zoom_level_ui)

    bg_c = (
        color_configs.get("background", rgb_to_hex(BG_COLOR))
        if color_configs
        else rgb_to_hex(BG_COLOR)
    )
    try:
        font_large = ImageFont.truetype(font_path, FONT_SIZE_LARGE)
        font_medium = ImageFont.truetype(font_path, FONT_SIZE_MEDIUM)
    except IOError:  # pragma: no cover - fallback
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()

    def compute_bearings(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        b = np.zeros(len(latitudes))
        for i in range(len(latitudes) - 1):
            lat1, lon1 = map(math.radians, (latitudes[i], longitudes[i]))
            lat2, lon2 = map(math.radians, (latitudes[i + 1], longitudes[i + 1]))
            dlon = lon2 - lon1
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            b[i] = (math.degrees(math.atan2(y, x)) + 360) % 360
        if len(b) >= 2:
            b[-1] = b[-2]
        return b

    bearings = compute_bearings(track["interp_lats"], track["interp_lons"])

    # Precompute graph coordinates
    graphs = {}
    if element_configs.get("Profil Altitude", {}).get("visible"):
        cfg = element_configs["Profil Altitude"]
        amin, amax = float(np.min(track["interp_eles"])), float(
            np.max(track["interp_eles"])
        )
        trans = GraphTransformer(amin, amax, cfg)
        coords_alt = [
            trans.to_xy(i, v, total_frames)
            for i, v in enumerate(track["interp_eles"])
        ]
        graphs["Profil Altitude"] = (coords_alt, amin, amax)

    if element_configs.get("Profil Vitesse", {}).get("visible"):
        cfg = element_configs["Profil Vitesse"]
        smin, smax = auto_speed_bounds(track["interp_speeds"])
        trans = GraphTransformer(smin, smax, cfg)
        coords_speed = [
            trans.to_xy(i, v, total_frames)
            for i, v in enumerate(track["interp_speeds"])
        ]
        graphs["Profil Vitesse"] = (coords_speed, smin, smax)

    if element_configs.get("Profil Allure", {}).get("visible"):
        cfg = element_configs["Profil Allure"]
        pace_vals = track["interp_pace"]
        finite = pace_vals[np.isfinite(pace_vals)]
        if finite.size:
            pmin, pmax = float(np.min(finite)), float(np.max(finite))
        else:
            pmin, pmax = 0.0, 10.0
        trans = GraphTransformer(pmin, pmax, cfg)
        coords_pace = [trans.to_xy(i, v, total_frames) for i, v in enumerate(pace_vals)]
        graphs["Profil Allure"] = (coords_pace, pmin, pmax)

    if element_configs.get("Profil Cardio", {}).get("visible"):
        cfg = element_configs["Profil Cardio"]
        hr_vals = track["interp_hrs"]
        finite = hr_vals[np.isfinite(hr_vals)]
        if finite.size:
            hmin, hmax = float(np.min(finite)), float(np.max(finite))
        else:
            hmin, hmax = 0.0, 200.0
        trans = GraphTransformer(hmin, hmax, cfg)
        coords_hr = [trans.to_xy(i, v, total_frames) for i, v in enumerate(hr_vals)]
        graphs["Profil Cardio"] = (coords_hr, hmin, hmax)

    # Video writing
    writer = imageio.get_writer(output_filename, fps=fps)

    for i in range(total_frames):
        img = Image.new("RGB", resolution, hex_to_rgb(bg_c))
        draw = ImageDraw.Draw(img)

        # Map rendering
        if (
            map_cfg.get("visible")
            and map_style in MAP_TILE_SERVERS
            and MAP_TILE_SERVERS[map_style]
            and STATICMAP_AVAILABLE
        ):
            sm = make_static_map(map_width, map_height, MAP_TILE_SERVERS[map_style])
            sm.add_line(Line(coords, PATH_COLOR, 4))
            sm.add_line(Line(coords[: i + 1], CURRENT_PATH_COLOR, 5))
            sm.add_marker(CircleMarker(coords[i], CURRENT_POINT_COLOR, 8))
            center = coords[i] if not fixed_map_view else None
            map_img = sm.render(zoom=zoom, center=center)
            img.paste(map_img, (map_cfg.get("x", 0), map_cfg.get("y", 0)))
            draw_north_arrow(
                img,
                {
                    "x": map_cfg.get("x", 0),
                    "y": map_cfg.get("y", 0),
                    "width": map_width,
                    "height": map_height,
                },
                bearings[i],
                TEXT_COLOR,
            )

        # Graphs
        for name, (coords_arr, vmin, vmax) in graphs.items():
            cfg = element_configs[name]
            draw_graph(
                draw,
                coords_arr,
                i,
                vmin,
                vmax,
                cfg,
                font_medium,
                name.split()[-1],
                "m" if "Altitude" in name else (
                    "km/h" if "Vitesse" in name else (
                        "min/km" if "Allure" in name else "bpm"
                    )
                ),
                PATH_COLOR,
                CURRENT_POINT_COLOR,
                TEXT_COLOR,
            )

        # Speed gauge
        if element_configs.get("Jauge Vitesse", {}).get("visible"):
            cfg = element_configs["Jauge Vitesse"]
            draw_circular_speedometer(
                draw,
                float(track["interp_speeds"][i]),
                graphs.get("Profil Vitesse", (None, 0.0, 80.0))[1],
                graphs.get("Profil Vitesse", (None, 0.0, 80.0))[2],
                cfg,
                font_medium,
                GAUGE_BG_COLOR,
                TEXT_COLOR,
            )

        # Text info and pace/HR
        if element_configs.get("Infos Texte", {}).get("visible"):
            cfg = element_configs["Infos Texte"]
            current_time = start_dt + timedelta(seconds=float(track["interp_times"][i]))
            draw_info_text(
                draw,
                float(track["interp_speeds"][i]),
                float(track["interp_eles"][i]),
                float(track["interp_slopes"][i]),
                current_time,
                cfg,
                font_large,
                tz,
                TEXT_COLOR,
            )
            draw_pace_hr_text(
                draw,
                float(track["interp_pace"][i]),
                float(track["interp_hrs"][i]),
                cfg,
                font_large,
                TEXT_COLOR,
            )

        writer.append_data(np.array(img))
        if progress_callback:
            progress_callback((i + 1) / total_frames)

    writer.close()
    return True


def render_first_frame_image(*args, **kwargs):
    """Placeholder that proxies to generate_preview_image."""
    resolution = kwargs.get("resolution", (1920, 1080))
    font_path = kwargs.get("font_path", "arial.ttf")
    element_configs = kwargs.get("element_configs", {})
    color_configs = kwargs.get("color_configs")
    return generate_preview_image(resolution, font_path, element_configs, color_configs)


__all__ = [
    "rgb_to_hex",
    "hex_to_rgb",
    "darken_color",
    "auto_speed_bounds",
    "GraphTransformer",
    "draw_graph",
    "draw_circular_speedometer",
    "draw_info_text",
    "draw_pace_hr_text",
    "draw_north_arrow",
    "generate_preview_image",
    "generate_gpx_video",
    "render_first_frame_image",
]
