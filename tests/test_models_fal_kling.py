"""Tests for fal.ai Kling v3 Pro cloud video models."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.models import GenerationRequest, ModelMode
from app.models.fal_kling import FalKlingImageToVideo, FalKlingTextToVideo


@pytest.fixture
def t2v_model():
    return FalKlingTextToVideo()


@pytest.fixture
def i2v_model():
    return FalKlingImageToVideo()


# --- Property tests ---


def test_t2v_properties(t2v_model):
    assert t2v_model.name == "Cloud: Kling v3"
    assert t2v_model.is_local is False
    assert t2v_model.is_loaded is True
    assert ModelMode.TEXT_TO_VIDEO in t2v_model.supported_modes


def test_i2v_properties(i2v_model):
    assert i2v_model.name == "Cloud: Kling v3 I2V"
    assert i2v_model.is_local is False
    assert i2v_model.is_loaded is True
    assert ModelMode.IMAGE_TO_VIDEO in i2v_model.supported_modes


# --- Validation tests ---


@pytest.mark.asyncio
async def test_t2v_requires_prompt(t2v_model):
    req = GenerationRequest(prompt="")
    with pytest.raises(ValueError, match="requires a text prompt"):
        await t2v_model.generate(req)


@pytest.mark.asyncio
async def test_i2v_requires_image(i2v_model):
    req = GenerationRequest(prompt="test", image=None)
    with pytest.raises(ValueError, match="requires an input image"):
        await i2v_model.generate(req)


# --- Generate T2V success ---


@pytest.mark.asyncio
async def test_t2v_generate_success(t2v_model, tmp_path):
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {"request_id": "kling-req-1"}
    mock_submit.raise_for_status = MagicMock()

    mock_status = MagicMock()
    mock_status.status_code = 200
    mock_status.json.return_value = {"status": "COMPLETED"}
    mock_status.raise_for_status = MagicMock()

    mock_result = MagicMock()
    mock_result.status_code = 200
    mock_result.json.return_value = {
        "video": {"url": "https://fal.ai/output/kling.mp4", "duration": 5.0}
    }
    mock_result.raise_for_status = MagicMock()

    mock_download = MagicMock()
    mock_download.status_code = 200
    mock_download.content = b"fake_video_data"
    mock_download.raise_for_status = MagicMock()

    with (
        patch.object(t2v_model, "_get_client") as mock_client_fn,
        patch("app.models.fal_base.settings") as mock_settings,
        patch("app.models.fal_base.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        mock_client.get = AsyncMock(side_effect=[mock_status, mock_result, mock_download])
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="a cat walking on the moon")
        result = await t2v_model.generate(req)

        assert result.model_name == "Cloud: Kling v3"
        assert result.mode == ModelMode.TEXT_TO_VIDEO
        assert result.video_path.exists()
        assert result.metadata["prompt"] == "a cat walking on the moon"
        assert result.metadata["video_duration"] == 5.0

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["prompt"] == "a cat walking on the moon"
        assert payload["duration"] == "5"
        assert payload["aspect_ratio"] == "16:9"


# --- Generate I2V success ---


@pytest.mark.asyncio
async def test_i2v_generate_success(i2v_model, tmp_path):
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {"request_id": "kling-req-2"}
    mock_submit.raise_for_status = MagicMock()

    mock_status = MagicMock()
    mock_status.status_code = 200
    mock_status.json.return_value = {"status": "COMPLETED"}
    mock_status.raise_for_status = MagicMock()

    mock_result = MagicMock()
    mock_result.status_code = 200
    mock_result.json.return_value = {"video": {"url": "https://fal.ai/output/kling_i2v.mp4"}}
    mock_result.raise_for_status = MagicMock()

    mock_download = MagicMock()
    mock_download.status_code = 200
    mock_download.content = b"fake_video_data"
    mock_download.raise_for_status = MagicMock()

    with (
        patch.object(i2v_model, "_get_client") as mock_client_fn,
        patch("app.models.fal_base.settings") as mock_settings,
        patch("app.models.fal_base.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        mock_client.get = AsyncMock(side_effect=[mock_status, mock_result, mock_download])
        mock_client_fn.return_value = mock_client

        test_image = Image.new("RGB", (64, 64), color="blue")
        req = GenerationRequest(prompt="animate this scene", image=test_image)
        result = await i2v_model.generate(req)

        assert result.model_name == "Cloud: Kling v3 I2V"
        assert result.mode == ModelMode.IMAGE_TO_VIDEO
        assert result.video_path.exists()

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "start_image_url" in payload
        assert payload["start_image_url"].startswith("data:image/png;base64,")
        assert payload["duration"] == "5"


# --- Polling tests ---


@pytest.mark.asyncio
async def test_t2v_no_request_id_raises(t2v_model):
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {}
    mock_submit.raise_for_status = MagicMock()

    with patch.object(t2v_model, "_get_client") as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="test")
        with pytest.raises(RuntimeError, match="request_id"):
            await t2v_model.generate(req)


@pytest.mark.asyncio
async def test_t2v_poll_failure(t2v_model, tmp_path):
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {"request_id": "kling-fail"}
    mock_submit.raise_for_status = MagicMock()

    mock_status = MagicMock()
    mock_status.status_code = 200
    mock_status.json.return_value = {"status": "FAILED", "error": "Content policy violation"}
    mock_status.raise_for_status = MagicMock()

    with (
        patch.object(t2v_model, "_get_client") as mock_client_fn,
        patch("app.models.fal_base.settings") as mock_settings,
        patch("app.models.fal_base.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        mock_client.get = AsyncMock(return_value=mock_status)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="should fail")
        with pytest.raises(RuntimeError, match="Content policy violation"):
            await t2v_model.generate(req)


# --- Health check tests ---


@pytest.mark.asyncio
async def test_t2v_health_check_success(t2v_model):
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch.object(t2v_model, "_get_client") as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_fn.return_value = mock_client

        result = await t2v_model.health_check()
        assert result["status"] == "ready"
        assert result["type"] == "cloud"


@pytest.mark.asyncio
async def test_t2v_health_check_error(t2v_model):
    result = await t2v_model.health_check()
    assert result["status"] == "error"
    assert result["model"] == "Cloud: Kling v3"
