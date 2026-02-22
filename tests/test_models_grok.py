"""Tests for Grok Imagine (xAI) cloud video model."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.models import GenerationRequest, ModelMode
from app.models.grok import GrokVideoModel


@pytest.fixture
def grok_model():
    return GrokVideoModel()


# --- Property tests ---


def test_grok_properties(grok_model):
    assert grok_model.name == "Cloud: Grok Video"
    assert grok_model.is_local is False
    assert grok_model.is_loaded is True
    assert ModelMode.TEXT_TO_VIDEO in grok_model.supported_modes
    assert ModelMode.IMAGE_TO_VIDEO in grok_model.supported_modes


# --- Validation tests ---


@pytest.mark.asyncio
async def test_grok_requires_prompt_or_image(grok_model):
    req = GenerationRequest(prompt="", image=None)
    with pytest.raises(ValueError, match="requires a text prompt or an input image"):
        await grok_model.generate(req)


# --- Generate T2V success ---


@pytest.mark.asyncio
async def test_grok_generate_t2v_success(grok_model, tmp_path):
    # Submit response
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {"request_id": "req-abc"}
    mock_submit.raise_for_status = MagicMock()

    # Poll response (done)
    mock_poll = MagicMock()
    mock_poll.status_code = 200
    mock_poll.json.return_value = {
        "status": "done",
        "video": {"url": "https://example.com/video.mp4"},
    }
    mock_poll.raise_for_status = MagicMock()

    # Download response
    mock_download = MagicMock()
    mock_download.status_code = 200
    mock_download.content = b"fake_video_data"
    mock_download.raise_for_status = MagicMock()

    with (
        patch.object(grok_model, "_get_client") as mock_client_fn,
        patch("app.models.grok.settings") as mock_settings,
        patch("app.models.grok.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        # First GET = poll, second GET = download
        mock_client.get = AsyncMock(side_effect=[mock_poll, mock_download])
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="a cat walking")
        result = await grok_model.generate(req)

        assert result.model_name == "Cloud: Grok Video"
        assert result.mode == ModelMode.TEXT_TO_VIDEO
        assert result.video_path.exists()
        assert result.metadata["prompt"] == "a cat walking"


# --- Generate I2V success ---


@pytest.mark.asyncio
async def test_grok_generate_i2v_success(grok_model, tmp_path):
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {"request_id": "req-xyz"}
    mock_submit.raise_for_status = MagicMock()

    mock_poll = MagicMock()
    mock_poll.status_code = 200
    mock_poll.json.return_value = {
        "status": "done",
        "video": {"url": "https://example.com/video.mp4"},
    }
    mock_poll.raise_for_status = MagicMock()

    mock_download = MagicMock()
    mock_download.status_code = 200
    mock_download.content = b"fake_video_data"
    mock_download.raise_for_status = MagicMock()

    with (
        patch.object(grok_model, "_get_client") as mock_client_fn,
        patch("app.models.grok.settings") as mock_settings,
        patch("app.models.grok.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        mock_client.get = AsyncMock(side_effect=[mock_poll, mock_download])
        mock_client_fn.return_value = mock_client

        test_image = Image.new("RGB", (64, 64), color="blue")
        req = GenerationRequest(prompt="animate this", image=test_image)
        result = await grok_model.generate(req)

        assert result.model_name == "Cloud: Grok Video"
        assert result.mode == ModelMode.IMAGE_TO_VIDEO
        assert result.metadata["mode"] == "image-to-video"

        # Verify image_url was sent in payload
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "image_url" in payload
        assert payload["image_url"].startswith("data:image/png;base64,")


# --- Polling tests ---


@pytest.mark.asyncio
async def test_grok_no_request_id_raises(grok_model):
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {}  # no request_id
    mock_submit.raise_for_status = MagicMock()

    with patch.object(grok_model, "_get_client") as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="test")
        with pytest.raises(RuntimeError, match="request_id"):
            await grok_model.generate(req)


@pytest.mark.asyncio
async def test_grok_poll_failure(grok_model, tmp_path):
    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = {"request_id": "req-fail"}
    mock_submit.raise_for_status = MagicMock()

    mock_poll = MagicMock()
    mock_poll.status_code = 200
    mock_poll.json.return_value = {"status": "failed", "error": "GPU error"}
    mock_poll.raise_for_status = MagicMock()

    with (
        patch.object(grok_model, "_get_client") as mock_client_fn,
        patch("app.models.grok.settings") as mock_settings,
        patch("app.models.grok.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.outputs_dir = tmp_path
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_submit)
        mock_client.get = AsyncMock(return_value=mock_poll)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="should fail")
        with pytest.raises(RuntimeError, match="GPU error"):
            await grok_model.generate(req)


# --- Error tests ---


@pytest.mark.asyncio
async def test_grok_http_error(grok_model):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

    with patch.object(grok_model, "_get_client") as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_fn.return_value = mock_client

        req = GenerationRequest(prompt="should fail")
        with pytest.raises(Exception, match="401"):
            await grok_model.generate(req)


# --- Health check tests ---


@pytest.mark.asyncio
async def test_grok_health_check_success(grok_model):
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch.object(grok_model, "_get_client") as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_fn.return_value = mock_client

        result = await grok_model.health_check()
        assert result["status"] == "ready"
        assert result["type"] == "cloud"


@pytest.mark.asyncio
async def test_grok_health_check_error(grok_model):
    result = await grok_model.health_check()
    assert result["status"] == "error"
    assert result["model"] == "Cloud: Grok Video"
