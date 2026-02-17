"""Output file management with auto-cleanup."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def get_output_files() -> list[Path]:
    """Return output video files sorted by modification time (newest first)."""
    outputs_dir = settings.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        outputs_dir.glob("*.mp4"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return files


def get_total_size_gb(files: list[Path]) -> float:
    """Calculate total size of files in GB."""
    return sum(f.stat().st_size for f in files) / (1024**3)


def cleanup_old_outputs() -> int:
    """Remove oldest files if count or size limits are exceeded. Returns number removed."""
    files = get_output_files()
    removed = 0

    # Remove by count
    while len(files) > settings.max_output_files:
        oldest = files.pop()
        oldest.unlink()
        removed += 1
        logger.info("Removed old output: %s", oldest.name)

    # Remove by size
    while files and get_total_size_gb(files) > settings.max_output_size_gb:
        oldest = files.pop()
        oldest.unlink()
        removed += 1
        logger.info("Removed output (size limit): %s", oldest.name)

    return removed


def get_gallery_items() -> list[str]:
    """Return file paths as strings for Gradio gallery."""
    return [str(f) for f in get_output_files()]
