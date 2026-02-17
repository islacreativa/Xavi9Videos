"""Video utilities using FFmpeg."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_video_info(path: Path) -> dict:
    """Get video metadata using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"error": "ffprobe failed"}

        data = json.loads(result.stdout)
        video_stream = next(
            (s for s in data.get("streams", []) if s["codec_type"] == "video"),
            {},
        )

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "duration": float(data.get("format", {}).get("duration", 0)),
            "fps": _parse_fps(video_stream.get("r_frame_rate", "0/1")),
            "codec": video_stream.get("codec_name", "unknown"),
            "size_mb": int(data.get("format", {}).get("size", 0)) / (1024 * 1024),
        }
    except Exception as e:
        logger.warning("Could not get video info for %s: %s", path, e)
        return {"error": str(e)}


def generate_thumbnail(video_path: Path, output_path: Path | None = None) -> Path | None:
    """Extract first frame as thumbnail."""
    if output_path is None:
        output_path = video_path.with_suffix(".jpg")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(output_path),
            ],
            capture_output=True,
            timeout=10,
        )
        return output_path if output_path.exists() else None
    except Exception as e:
        logger.warning("Could not generate thumbnail: %s", e)
        return None


def _parse_fps(rate_str: str) -> float:
    """Parse ffprobe frame rate string like '24/1' or '25'."""
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/")
            return round(int(num) / int(den), 2)
        return float(rate_str)
    except (ValueError, ZeroDivisionError):
        return 0.0
