"""Unit tests for the Wan 2.1 model wrapper."""

import pytest

from app.models import GenerationRequest, ModelMode
from app.models.wan import WanModel


def test_properties():
    model = WanModel()
    assert model.name == "Wan 2.1"
    assert ModelMode.TEXT_TO_VIDEO in model.supported_modes
    assert ModelMode.IMAGE_TO_VIDEO in model.supported_modes
    assert model.is_local is True
    assert model.is_loaded is False


@pytest.mark.asyncio
async def test_generate_not_loaded():
    model = WanModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        await model.generate(GenerationRequest(prompt="test"))


@pytest.mark.asyncio
async def test_unload_when_not_loaded():
    model = WanModel()
    await model.unload()  # should be a no-op
    assert model.is_loaded is False


@pytest.mark.asyncio
async def test_health_not_loaded():
    model = WanModel()
    result = await model.health_check()
    assert result["status"] == "available"
    assert result["model"] == "Wan 2.1"
    assert result["loaded"] is False
