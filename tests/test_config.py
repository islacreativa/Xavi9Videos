"""Tests for app.config."""

from pathlib import Path

from app.config import Settings


def test_default_settings():
    s = Settings(
        cosmos_nim_url="http://test:8000",
        models_dir=Path("/tmp/models"),
        outputs_dir=Path("/tmp/outputs"),
    )
    assert s.cosmos_nim_url == "http://test:8000"
    assert s.ltx2_use_fp8 is True
    assert s.max_concurrent_requests == 1
    assert s.default_num_frames == 49


def test_local_paths():
    s = Settings(models_dir=Path("/tmp/models"))
    assert s.ltx2_local_path == Path("/tmp/models/ltx2")
    assert s.svd_local_path == Path("/tmp/models/svd")
