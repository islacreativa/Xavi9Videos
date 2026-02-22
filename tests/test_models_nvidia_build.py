"""Tests for NVIDIA Build API cloud models."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.models import GenerationRequest, ModelMode
from app.models.nvidia_build import NvidiaBuildText2World, NvidiaBuildVideo2World


@pytest.fixture
def text2world():
    return NvidiaBuildText2World()


@pytest.fixture
def video2world():
    return NvidiaBuildVideo2World()


# --- Property tests ---


def test_text2world_properties(text2world):
    assert text2world.name == "Cloud: Cosmos Text2World"
    assert text2world.is_local is False
    assert text2world.is_loaded is True
    assert ModelMode.TEXT_TO_VIDEO in text2world.supported_modes


def test_video2world_properties(video2world):
    assert video2world.name == "Cloud: Cosmos Video2World"
    assert video2world.is_local is False
    assert video2world.is_loaded is True
    assert ModelMode.IMAGE_TO_VIDEO in video2world.supported_modes


# --- Validation tests ---


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


# --- Health check tests ---


@pytest.mark.asyncio
async def test_health_check_success(text2world):
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch.object(text2world, "_get_client") as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_fn.return_value = mock_client

        result = await text2world.health_check()
        assert result["status"] == "ready"
        assert result["type"] == "cloud"


@pytest.mark.asyncio
async def test_health_check_error(text2world):
    result = await text2world.health_check()
    assert result["status"] == "error"
    assert result["model"] == "Cloud: Cosmos Text2World"


# --- Generate success tests ---


@pytest.mark.asyncio
async def test_text2world_generate_success(text2world, tmp_path):
    fake_video = base64.b64encode(b"fake_video_data").decode()
    api_response = {
        "b64_video": fake_video,
        "seed": 42,
        "upsampled_prompt": "enhanced prompt",
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = api_response
    mock_response.raise_for_status = MagicMock()

    with (
        patch.object(text2world, "_get_client") as mock_client_fn,
        patch("app.models.nvidia_build.settings") as mock_settings,
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="a cat walking", num_frames=25, fps=24)
        result = await text2world.generate(req)

        assert result.model_name == "Cloud: Cosmos Text2World"
        assert result.mode == ModelMode.TEXT_TO_VIDEO
        assert result.video_path.exists()
        assert result.metadata["seed"] == 42
        assert result.metadata["upsampled_prompt"] == "enhanced prompt"

        # Verify payload format
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json") or call_args[0][1]
        assert "video_params" in payload
        assert payload["video_params"]["frames_count"] == 25
        assert payload["video_params"]["frames_per_sec"] == 24


@pytest.mark.asyncio
async def test_video2world_generate_success(video2world, tmp_path):
    fake_video = base64.b64encode(b"fake_video_data").decode()
    api_response = {"b64_video": fake_video, "seed": 99}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = api_response
    mock_response.raise_for_status = MagicMock()

    with (
        patch.object(video2world, "_get_client") as mock_client_fn,
        patch("app.models.nvidia_build.settings") as mock_settings,
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_fn.return_value = mock_client

        test_image = Image.new("RGB", (64, 64), color="red")
        req = GenerationRequest(prompt="expand world", image=test_image)
        result = await video2world.generate(req)

        assert result.model_name == "Cloud: Cosmos Video2World"
        assert result.mode == ModelMode.IMAGE_TO_VIDEO
        assert result.video_path.exists()
        assert result.metadata["seed"] == 99

        # Verify image was base64-encoded in payload
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json") or call_args[0][1]
        assert "image" in payload
        assert "video_params" in payload


# --- Polling tests ---


@pytest.mark.asyncio
async def test_text2world_generate_with_polling(text2world, tmp_path):
    """Test async polling when API returns HTTP 202."""
    fake_video = base64.b64encode(b"polled_video").decode()

    # Initial 202 response
    mock_202 = MagicMock()
    mock_202.status_code = 202
    mock_202.headers = {"NVCF-REQID": "req-123"}
    mock_202.raise_for_status = MagicMock()

    # Poll 200 response
    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {"b64_video": fake_video, "seed": 7}

    with (
        patch.object(text2world, "_get_client") as mock_client_fn,
        patch("app.models.nvidia_build.settings") as mock_settings,
        patch("app.models.nvidia_build.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_202)
        mock_client.get = AsyncMock(return_value=mock_200)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="a polled video")
        result = await text2world.generate(req)

        assert result.video_path.exists()
        mock_client.get.assert_called()


# --- Error tests ---


@pytest.mark.asyncio
async def test_generate_http_error(text2world):
    """Test that HTTP errors are raised properly."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

    with patch.object(text2world, "_get_client") as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="should fail")
        with pytest.raises(Exception, match="401"):
            await text2world.generate(req)


@pytest.mark.asyncio
async def test_polling_no_reqid_raises(text2world):
    """Test that 202 without NVCF-REQID raises RuntimeError."""
    mock_202 = MagicMock()
    mock_202.status_code = 202
    mock_202.headers = {}  # no NVCF-REQID

    with patch.object(text2world, "_get_client") as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_202)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="no reqid")
        with pytest.raises(RuntimeError, match="NVCF-REQID"):
            await text2world.generate(req)
