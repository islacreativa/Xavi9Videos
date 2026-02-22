"""Integration tests for the orchestrator (app.main)."""

import pytest

from app.main import check_health, generate_video


@pytest.mark.asyncio
async def test_unknown_model():
    video, meta, status = await generate_video(
        "NonExistent", "prompt", None, 768, 512, 49, 24, 30, 3.0, -1
    )
    assert video is None
    assert "Unknown model" in status


@pytest.mark.asyncio
async def test_text_model_no_prompt():
    # Use LTX-2 since Cosmos may not be registered (no NIM URL)
    video, meta, status = await generate_video("LTX-2", "", None, 768, 512, 49, 24, 30, 3.0, -1)
    assert video is None
    assert "prompt" in status.lower()


@pytest.mark.asyncio
async def test_image_model_no_image():
    video, meta, status = await generate_video("SVD-XT", "", None, 768, 512, 49, 24, 30, 3.0, -1)
    assert video is None
    assert "requires" in status.lower() or "image" in status.lower()


@pytest.mark.asyncio
async def test_check_health_returns_local_models():
    result = await check_health()
    # Local models are always registered
    assert "LTX-2" in result
    assert "SVD-XT" in result
    assert "Wan 2.1" in result


@pytest.mark.asyncio
async def test_check_health_cosmos_conditional():
    """Cosmos models only present when COSMOS_NIM_URL is set."""
    from app.config import settings

    if settings.cosmos_nim_url:
        result = await check_health()
        assert "Cosmos Text2World" in result
        assert "Cosmos Video2World" in result
