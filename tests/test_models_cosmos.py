"""Tests for Cosmos NIM client models."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from PIL import Image

from app.models import GenerationRequest
from app.models.cosmos import CosmosText2World, CosmosVideo2World


@pytest.fixture
def text2world():
    return CosmosText2World()


@pytest.fixture
def video2world():
    return CosmosVideo2World()


@pytest.mark.asyncio
async def test_text2world_requires_prompt(text2world):
    req = GenerationRequest(prompt="")
    with pytest.raises(ValueError, match="requires a text prompt"):
        await text2world.generate(req)


@pytest.mark.asyncio
async def test_video2world_requires_image(video2world):
    req = GenerationRequest(prompt="test", image=None)
    with pytest.raises(ValueError, match="requires an input image"):
        await video2world.generate(req)


@pytest.mark.asyncio
async def test_text2world_health_error(text2world):
    result = await text2world.health_check()
    assert result["status"] == "error"
    assert result["model"] == "Cosmos Text2World"


@pytest.mark.asyncio
async def test_text2world_properties(text2world):
    assert text2world.name == "Cosmos Text2World"
    assert text2world.is_local is False
    assert text2world.is_loaded is True


@pytest.mark.asyncio
async def test_video2world_properties(video2world):
    assert video2world.name == "Cosmos Video2World"
    assert video2world.is_local is False


@pytest.mark.asyncio
async def test_text2world_generate_success(text2world, tmp_path):
    fake_video = base64.b64encode(b"fake_video_data").decode()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"video": fake_video}
    mock_response.raise_for_status = MagicMock()

    with patch.object(text2world, "_get_client") as mock_client_fn, \
         patch("app.models.cosmos.settings") as mock_settings:
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="a cat walking")
        result = await text2world.generate(req)

        assert result.model_name == "Cosmos Text2World"
        assert result.video_path.exists()
