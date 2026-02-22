"""Tests for app.utils.storage."""

import time
from unittest.mock import patch

from app.utils.storage import cleanup_old_outputs, get_gallery_items, get_output_files


def test_get_output_files(tmp_path):
    with patch("app.utils.storage.settings") as mock_settings:
        mock_settings.outputs_dir = tmp_path

        # Create test files
        (tmp_path / "a.mp4").write_bytes(b"a")
        time.sleep(0.01)
        (tmp_path / "b.mp4").write_bytes(b"bb")

        files = get_output_files()
        assert len(files) == 2
        assert files[0].name == "b.mp4"  # newest first


def test_cleanup_by_count(tmp_path):
    with patch("app.utils.storage.settings") as mock_settings:
        mock_settings.outputs_dir = tmp_path
        mock_settings.max_output_files = 2
        mock_settings.max_output_size_gb = 100.0

        for i in range(5):
            (tmp_path / f"vid_{i}.mp4").write_bytes(b"x" * 100)
            time.sleep(0.01)

        removed = cleanup_old_outputs()
        assert removed == 3
        remaining = list(tmp_path.glob("*.mp4"))
        assert len(remaining) == 2


def test_get_gallery_items(tmp_path):
    with patch("app.utils.storage.settings") as mock_settings:
        mock_settings.outputs_dir = tmp_path

        (tmp_path / "test.mp4").write_bytes(b"data")
        items = get_gallery_items()
        assert len(items) == 1
        assert items[0].endswith("test.mp4")
