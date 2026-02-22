"""Tests for fal.ai LTX-2 Pro cloud video models."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.models import GenerationRequest, ModelMode
from app.models.fal_ltx2 import FalLTX2ImageToVideo, FalLTX2TextToVideo


@pytest.fixture
def t2v_model():
    return FalLTX2TextToVideo()


@pytest.fixture
def i2v_model():
    return FalLTX2ImageToVideo()


# --- Property tests ---


def test_t2v_properties(t2v_model):
    assert t2v_model.name == "Cloud: LTX-2 Pro"
    assert t2v_model.is_local is False
    assert t2v_model.is_loaded is True
    assert ModelMode.TEXT_TO_VIDEO in t2v_model.supported_modes


def test_i2v_properties(i2v_model):
    assert i2v_model.name == "Cloud: LTX-2 Pro I2V"
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
    # Submit response
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {"request_id": "fal-req-1"}
    mock_submit.raise_for_status = MagicMock()

    # Poll status response
    mock_status = MagicMock()
    mock_status.status_code = 200
    mock_status.json.return_value = {"status": "COMPLETED"}
    mock_status.raise_for_status = MagicMock()

    # Result response
    mock_result = MagicMock()
    mock_result.status_code = 200
    mock_result.json.return_value = {
        "video": {"url": "https://fal.ai/output/video.mp4", "duration": 2.0}
    }
    mock_result.raise_for_status = MagicMock()

    # Download response
    mock_download = MagicMock()
    mock_download.status_code = 200
    mock_download.content = b"fake_video_data"
    mock_download.raise_for_status = MagicMock()

    with (
        patch.object(t2v_model, "_get_client") as mock_client_fn,
        patch("app.models.fal_ltx2.settings") as mock_settings,
        patch("app.models.fal_ltx2.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        # GET calls: poll status, fetch result, download video
        mock_client.get = AsyncMock(side_effect=[mock_status, mock_result, mock_download])
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="a sunset over mountains", num_frames=25)
        result = await t2v_model.generate(req)

        assert result.model_name == "Cloud: LTX-2 Pro"
        assert result.mode == ModelMode.TEXT_TO_VIDEO
        assert result.video_path.exists()
        assert result.metadata["prompt"] == "a sunset over mountains"
        assert result.metadata["video_duration"] == 2.0

        # Verify payload
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["prompt"] == "a sunset over mountains"
        assert payload["num_frames"] == 25


# --- Generate I2V success ---


@pytest.mark.asyncio
async def test_i2v_generate_success(i2v_model, tmp_path):
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {"request_id": "fal-req-2"}
    mock_submit.raise_for_status = MagicMock()

    mock_status = MagicMock()
    mock_status.status_code = 200
    mock_status.json.return_value = {"status": "COMPLETED"}
    mock_status.raise_for_status = MagicMock()

    mock_result = MagicMock()
    mock_result.status_code = 200
    mock_result.json.return_value = {"video": {"url": "https://fal.ai/output/video.mp4"}}
    mock_result.raise_for_status = MagicMock()

    mock_download = MagicMock()
    mock_download.status_code = 200
    mock_download.content = b"fake_video_data"
    mock_download.raise_for_status = MagicMock()

    with (
        patch.object(i2v_model, "_get_client") as mock_client_fn,
        patch("app.models.fal_ltx2.settings") as mock_settings,
        patch("app.models.fal_ltx2.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        mock_client.get = AsyncMock(side_effect=[mock_status, mock_result, mock_download])
        mock_client_fn.return_value = mock_client

        test_image = Image.new("RGB", (64, 64), color="green")
        req = GenerationRequest(prompt="animate this", image=test_image)
        result = await i2v_model.generate(req)

        assert result.model_name == "Cloud: LTX-2 Pro I2V"
        assert result.mode == ModelMode.IMAGE_TO_VIDEO
        assert result.video_path.exists()

        # Verify image_url in payload
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "image_url" in payload
        assert payload["image_url"].startswith("data:image/png;base64,")


# --- Polling tests ---


@pytest.mark.asyncio
async def test_t2v_no_request_id_raises(t2v_model):
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {}  # no request_id
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
    mock_submit.json.return_value = {"request_id": "fal-fail"}
    mock_submit.raise_for_status = MagicMock()

    mock_status = MagicMock()
    mock_status.status_code = 200
    mock_status.json.return_value = {"status": "FAILED", "error": "Out of capacity"}
    mock_status.raise_for_status = MagicMock()

    with (
        patch.object(t2v_model, "_get_client") as mock_client_fn,
        patch("app.models.fal_ltx2.settings") as mock_settings,
        patch("app.models.fal_ltx2.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        mock_client.get = AsyncMock(return_value=mock_status)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="should fail")
        with pytest.raises(RuntimeError, match="Out of capacity"):
            await t2v_model.generate(req)


# --- Error tests ---


@pytest.mark.asyncio
async def test_t2v_http_error(t2v_model):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

    with patch.object(t2v_model, "_get_client") as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="should fail")
        with pytest.raises(Exception, match="401"):
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
    assert result["model"] == "Cloud: LTX-2 Pro"
