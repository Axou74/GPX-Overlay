from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import ImageFont

from OverlayGPX_V1 import DEFAULT_ELEMENT_CONFIGS, prepare_render_context
from rendering import load_interpolated_track


SAMPLE_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="pytest" xmlns="http://www.topografix.com/GPX/1/1" xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">
  <trk>
    <name>Test Track</name>
    <trkseg>
      <trkpt lat="45.0000" lon="6.0000">
        <ele>1200.0</ele>
        <time>2023-01-01T08:00:00Z</time>
        <extensions>
          <gpxtpx:TrackPointExtension>
            <gpxtpx:hr>120</gpxtpx:hr>
          </gpxtpx:TrackPointExtension>
        </extensions>
      </trkpt>
      <trkpt lat="45.0005" lon="6.0005">
        <ele>1210.0</ele>
        <time>2023-01-01T08:00:10Z</time>
        <extensions>
          <gpxtpx:TrackPointExtension>
            <gpxtpx:hr>122</gpxtpx:hr>
          </gpxtpx:TrackPointExtension>
        </extensions>
      </trkpt>
      <trkpt lat="45.0010" lon="6.0010">
        <ele>1220.0</ele>
        <time>2023-01-01T08:00:20Z</time>
        <extensions>
          <gpxtpx:TrackPointExtension>
            <gpxtpx:hr>125</gpxtpx:hr>
          </gpxtpx:TrackPointExtension>
        </extensions>
      </trkpt>
    </trkseg>
  </trk>
</gpx>
"""


def test_prepare_render_context_generates_consistent_paths(tmp_path: Path) -> None:
    gpx_file = tmp_path / "sample.gpx"
    gpx_file.write_text(SAMPLE_GPX, encoding="utf-8")

    track = load_interpolated_track(
        str(gpx_file),
        start_offset=0,
        clip_duration=30,
        fps=2,
    )

    element_configs = {name: copy.deepcopy(cfg) for name, cfg in DEFAULT_ELEMENT_CONFIGS.items()}

    colors = {
        "background": (0, 0, 0),
        "map_path": (200, 200, 200),
        "map_current_path": (255, 255, 255),
        "map_current_point": (255, 0, 0),
        "graph_altitude": (180, 180, 180),
        "graph_speed": (150, 150, 255),
        "graph_pace": (150, 255, 150),
        "graph_hr": (255, 150, 150),
        "graph_current_point": (255, 0, 0),
        "text": (255, 255, 255),
        "gauge_background": (30, 30, 30),
    }

    context = prepare_render_context(
        resolution=(800, 600),
        element_configs=element_configs,
        track=track,
        font_graph=ImageFont.load_default(),
        colors=colors,
    )

    assert context.map_area == element_configs["Carte"]
    assert len(context.graph_layers) >= 3
    assert all(len(layer["path"]) == track.total_frames for layer in context.graph_layers)

    lon_center = (track.lon_min_raw + track.lon_max_raw) * 0.5
    lat_center = (track.lat_min_raw + track.lat_max_raw) * 0.5
    assert context.map_center == pytest.approx((lon_center, lat_center))

    assert context.speed_bounds[1] >= context.speed_bounds[0]
