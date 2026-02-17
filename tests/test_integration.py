"""Integration tests for the orchestrator (app.main)."""

from unittest.mock import AsyncMock, patch

import pytest

from app.main import check_health, generate_video, models


@pytest.mark.asyncio
async def test_unknown_model():
    video, meta, status = await generate_video(
        "NonExistent", "prompt", None, 768, 512, 49, 24, 30, 3.0, -1
    )
    assert video is None
    assert "Unknown model" in status


@pytest.mark.asyncio
async def test_text_model_no_prompt():
    video, meta, status = await generate_video(
        "Cosmos Text2World", "", None, 768, 512, 49, 24, 30, 3.0, -1
    )
    assert video is None
    assert "prompt" in status.lower()


@pytest.mark.asyncio
async def test_image_model_no_image():
    video, meta, status = await generate_video(
        "SVD-XT", "", None, 768, 512, 49, 24, 30, 3.0, -1
    )
    assert video is None
    assert "requires" in status.lower() or "image" in status.lower()


@pytest.mark.asyncio
async def test_check_health_returns_all_models():
    result = await check_health()
    assert "Cosmos Text2World" in result
    assert "Cosmos Video2World" in result
    assert "LTX-2" in result
    assert "SVD-XT" in result
