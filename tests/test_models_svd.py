"""Tests for SVD-XT model."""

import pytest

from app.models import GenerationRequest, ModelMode
from app.models.svd import SVD_HEIGHT, SVD_NUM_FRAMES, SVD_WIDTH, SVDModel


@pytest.fixture
def svd():
    return SVDModel()


def test_properties(svd):
    assert svd.name == "SVD-XT"
    assert svd.is_local is True
    assert svd.is_loaded is False
    assert svd.supported_modes == [ModelMode.IMAGE_TO_VIDEO]


@pytest.mark.asyncio
async def test_generate_requires_image(svd):
    svd._loaded = True
    svd._pipeline = "fake"
    req = GenerationRequest(prompt="test", image=None)
    with pytest.raises(ValueError, match="requires an input image"):
        await svd.generate(req)


@pytest.mark.asyncio
async def test_generate_not_loaded(svd):
    from PIL import Image

    img = Image.new("RGB", (512, 512))
    req = GenerationRequest(image=img)
    with pytest.raises(RuntimeError, match="not loaded"):
        await svd.generate(req)


@pytest.mark.asyncio
async def test_unload_when_not_loaded(svd):
    await svd.unload()
    assert svd.is_loaded is False


@pytest.mark.asyncio
async def test_health_weights_missing(svd, tmp_path):
    from unittest.mock import patch

    with patch("app.models.svd.settings") as mock_settings:
        mock_settings.svd_local_path = tmp_path / "nonexistent"
        result = await svd.health_check()
        assert result["status"] == "weights_missing"
        assert result["loaded"] is False


def test_fixed_params():
    assert SVD_WIDTH == 1024
    assert SVD_HEIGHT == 576
    assert SVD_NUM_FRAMES == 25
