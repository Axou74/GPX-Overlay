"""Command line entry point for GPX overlay rendering."""
from __future__ import annotations

import argparse

from gpx_overlay import (
    DEFAULT_CLIP_DURATION_SECONDS,
    DEFAULT_FPS,
    DEFAULT_FONT_PATH,
    DEFAULT_RESOLUTION,
    DEFAULT_ELEMENT_CONFIGS,
)
from gpx_overlay.video_renderer import generate_gpx_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GPX overlay video")
    parser.add_argument("gpx", help="GPX input file")
    parser.add_argument("output", help="Output video file (mp4)")
    parser.add_argument("--duration", type=int, default=DEFAULT_CLIP_DURATION_SECONDS, help="Clip duration in seconds")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second")
    args = parser.parse_args()

    generate_gpx_video(
        args.gpx,
        args.output,
        start_offset=0,
        clip_duration=args.duration,
        fps=args.fps,
        resolution=DEFAULT_RESOLUTION,
        font_path=DEFAULT_FONT_PATH,
        element_configs=DEFAULT_ELEMENT_CONFIGS,
    )


if __name__ == "__main__":
    main()
