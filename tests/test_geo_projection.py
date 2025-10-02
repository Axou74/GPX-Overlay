"""Tests de validation pour les conversions WebMercator."""

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402  (import après le skip conditionnel)

from OverlayGPX_V1 import (  # noqa: E402
    MERCATOR_LAT_MAX,
    bbox_fit_zoom,
    lonlat_to_pixel,
    lonlat_to_pixel_np,
    pixel_to_lonlat,
    pixel_to_lonlat_np,
)


@pytest.mark.parametrize(
    "lon, lat",
    [
        (0.0, 0.0),
        (2.3522, 48.8566),  # Paris
        (-73.9857, 40.7484),  # New York
        (139.6917, 35.6895),  # Tokyo
        (12.4922, 41.8902),  # Rome
    ],
)
@pytest.mark.parametrize("zoom", [0, 5, 10, 15, 19])
def test_lonlat_pixel_round_trip(lon, lat, zoom):
    """Vérifie que lonlat_to_pixel et son inverse sont cohérents."""

    x, y = lonlat_to_pixel(lon, lat, zoom)
    lon_back, lat_back = pixel_to_lonlat(x, y, zoom)
    assert lon_back == pytest.approx(lon, rel=0, abs=1e-9)
    assert lat_back == pytest.approx(lat, rel=0, abs=1e-9)


def test_lonlat_clamping_on_poles():
    """Les latitudes hors domaine doivent être ramenées dans la plage valide."""

    lon, lat = 10.0, MERCATOR_LAT_MAX + 5.0
    x, y = lonlat_to_pixel(lon, lat, 10)
    lon_back, lat_back = pixel_to_lonlat(x, y, 10)
    assert lon_back == pytest.approx(lon, rel=0, abs=1e-9)
    assert lat_back <= MERCATOR_LAT_MAX


def test_vectorised_projection_matches_scalar():
    """La version vectorisée doit donner les mêmes résultats que la version scalaire."""

    lons = np.array([3.0, -1.2, 45.3, -122.33], dtype=float)
    lats = np.array([48.0, -33.9, 12.5, 47.61], dtype=float)
    zoom = 12

    xs_np, ys_np = lonlat_to_pixel_np(lons, lats, zoom)

    xs = []
    ys = []
    for lon, lat in zip(lons, lats):
        x, y = lonlat_to_pixel(float(lon), float(lat), zoom)
        xs.append(x)
        ys.append(y)

    assert np.allclose(xs_np, xs)
    assert np.allclose(ys_np, ys)

    lon_back, lat_back = pixel_to_lonlat_np(xs_np, ys_np, zoom)
    assert np.allclose(lon_back, lons)
    assert np.allclose(lat_back, np.clip(lats, -MERCATOR_LAT_MAX, MERCATOR_LAT_MAX))


def test_bbox_fit_zoom_contains_bbox():
    """Le zoom retourné doit contenir la boîte englobante demandée."""

    lon_min, lat_min = -1.0, 47.0
    lon_max, lat_max = -0.5, 47.3
    width, height = 800, 600

    zoom = bbox_fit_zoom(width, height, lon_min, lat_min, lon_max, lat_max)

    x1, y1 = lonlat_to_pixel(lon_min, lat_min, zoom)
    x2, y2 = lonlat_to_pixel(lon_max, lat_max, zoom)

    assert abs(x2 - x1) <= width
    assert abs(y2 - y1) <= height

    if zoom < 19:
        x1_next, y1_next = lonlat_to_pixel(lon_min, lat_min, zoom + 1)
        x2_next, y2_next = lonlat_to_pixel(lon_max, lat_max, zoom + 1)
        assert abs(x2_next - x1_next) > width or abs(y2_next - y1_next) > height
