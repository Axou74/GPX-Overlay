"""Core definitions and configuration for the GPX overlay package."""

# Map tile servers available for rendering backgrounds.
MAP_TILE_SERVERS = {
    "Aucun": None,
    "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "IGN Satellite": (
        "https://wxs.ign.fr/essentiels/geoportail/wmts?REQUEST=GetTile&"
        "SERVICE=WMTS&VERSION=1.0.0&STYLE=normal&FORMAT=image/jpeg&"
        "LAYER=ORTHOIMAGERY.ORTHOPHOTOS&TILEMATRIXSET=PM&TILEMATRIX={z}&"
        "TILEROW={y}&TILECOL={x}"
    ),
    "CyclOSM (FR)": "https://a.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png",
}

# Geometry and appearance defaults
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_FPS = 25
DEFAULT_FONT_PATH = "arial.ttf"
DEFAULT_CLIP_DURATION_SECONDS = 5

# Colour palette
BG_COLOR = (0, 0, 0)
PATH_COLOR = (200, 200, 200)
CURRENT_PATH_COLOR = (255, 255, 255)
CURRENT_POINT_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
GAUGE_BG_COLOR = (30, 30, 30)

# Layout constants
FONT_SIZE_LARGE = 40
FONT_SIZE_MEDIUM = 30
MARGIN = 50
GRAPH_PADDING = 100

DEFAULT_ELEMENT_CONFIGS = {
    "Carte": {
        "visible": True,
        "x": MARGIN,
        "y": MARGIN,
        "width": DEFAULT_RESOLUTION[0] // 2 - MARGIN * 2,
        "height": DEFAULT_RESOLUTION[1] // 2 - MARGIN * 2,
    },
    "Profil Altitude": {
        "visible": True,
        "x": DEFAULT_RESOLUTION[0] // 2 + MARGIN,
        "y": MARGIN,
        "width": DEFAULT_RESOLUTION[0] // 2 - MARGIN * 2,
        "height": 200,
    },
    "Profil Vitesse": {
        "visible": True,
        "x": DEFAULT_RESOLUTION[0] // 2 + MARGIN,
        "y": MARGIN + 200 + GRAPH_PADDING,
        "width": DEFAULT_RESOLUTION[0] // 2 - MARGIN * 2,
        "height": 150,
    },
    "Profil Allure": {
        "visible": True,
        "x": DEFAULT_RESOLUTION[0] // 2 + MARGIN,
        "y": MARGIN + 200 + GRAPH_PADDING + 150 + GRAPH_PADDING,
        "width": DEFAULT_RESOLUTION[0] // 2 - MARGIN * 2,
        "height": 150,
    },
    "Profil Cardio": {
        "visible": True,
        "x": DEFAULT_RESOLUTION[0] // 2 + MARGIN,
        "y": MARGIN + 200 + GRAPH_PADDING + 150 + GRAPH_PADDING + 150 + GRAPH_PADDING,
        "width": DEFAULT_RESOLUTION[0] // 2 - MARGIN * 2,
        "height": 150,
    },
    "Jauge Vitesse": {
        "visible": True,
        "x": MARGIN,
        "y": DEFAULT_RESOLUTION[1] - 100 - MARGIN,
        "width": 300,
        "height": 50,
    },
    "Infos Texte": {
        "visible": True,
        "x": MARGIN,
        "y": DEFAULT_RESOLUTION[1] - (6 * FONT_SIZE_LARGE + 50) - 50 - 2 * MARGIN,
        "width": 450,
        "height": 6 * FONT_SIZE_LARGE + 50,
    },
}

__all__ = [
    "MAP_TILE_SERVERS",
    "DEFAULT_RESOLUTION",
    "DEFAULT_FPS",
    "DEFAULT_FONT_PATH",
    "DEFAULT_CLIP_DURATION_SECONDS",
    "BG_COLOR",
    "PATH_COLOR",
    "CURRENT_PATH_COLOR",
    "CURRENT_POINT_COLOR",
    "TEXT_COLOR",
    "GAUGE_BG_COLOR",
    "FONT_SIZE_LARGE",
    "FONT_SIZE_MEDIUM",
    "MARGIN",
    "GRAPH_PADDING",
    "DEFAULT_ELEMENT_CONFIGS",
]
