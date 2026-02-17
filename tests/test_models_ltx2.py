"""Tests for LTX-2 model."""

import pytest

from app.models import GenerationRequest, ModelMode
from app.models.ltx2 import LTX2Model


@pytest.fixture
def ltx2():
    return LTX2Model()


def test_properties(ltx2):
    assert ltx2.name == "LTX-2"
    assert ltx2.is_local is True
    assert ltx2.is_loaded is False
    assert ModelMode.TEXT_TO_VIDEO in ltx2.supported_modes
    assert ModelMode.IMAGE_TO_VIDEO in ltx2.supported_modes


@pytest.mark.asyncio
async def test_generate_not_loaded(ltx2):
    req = GenerationRequest(prompt="test")
    with pytest.raises(RuntimeError, match="not loaded"):
        await ltx2.generate(req)


@pytest.mark.asyncio
async def test_unload_when_not_loaded(ltx2):
    await ltx2.unload()  # Should not raise
    assert ltx2.is_loaded is False


@pytest.mark.asyncio
async def test_health_weights_missing(ltx2, tmp_path):
    from unittest.mock import patch

    with patch("app.models.ltx2.settings") as mock_settings:
        mock_settings.ltx2_local_path = tmp_path / "nonexistent"
        result = await ltx2.health_check()
        assert result["status"] == "weights_missing"
        assert result["loaded"] is False
