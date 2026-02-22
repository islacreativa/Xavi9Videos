"""End-to-end GPU tests for real video generation.

Run with: pytest -m gpu tests/test_gpu.py -v
Requires: NVIDIA GPU with CUDA support and model weights downloaded.
"""

import asyncio

import numpy as np
import pytest
from PIL import Image

from app.models import GenerationRequest, ModelMode
from app.models.ltx2 import LTX2Model
from app.models.svd import SVDModel
from app.models.wan import WanModel

pytestmark = pytest.mark.gpu


@pytest.fixture
def ltx2_model():
    model = LTX2Model()
    yield model
    asyncio.get_event_loop().run_until_complete(model.unload())


@pytest.fixture
def svd_model():
    model = SVDModel()
    yield model
    asyncio.get_event_loop().run_until_complete(model.unload())


@pytest.fixture
def test_image():
    """Create a simple test image for image-to-video models."""
    return Image.fromarray(np.random.randint(0, 255, (576, 1024, 3), dtype=np.uint8))


class TestLTX2GPU:
    @pytest.mark.asyncio
    async def test_load_and_generate(self, ltx2_model):
        await ltx2_model.load()
        assert ltx2_model.is_loaded

        req = GenerationRequest(
            prompt="A cat sitting on a windowsill",
            width=512,
            height=320,
            num_frames=9,
            fps=24,
            guidance_scale=4.0,
            num_inference_steps=5,
            seed=42,
        )
        result = await ltx2_model.generate(req)

        assert result.video_path.exists()
        assert result.video_path.suffix == ".mp4"
        assert result.model_name == "LTX-2"
        assert result.mode == ModelMode.TEXT_TO_VIDEO
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_unload_frees_memory(self, ltx2_model):
        import torch

        await ltx2_model.load()
        assert ltx2_model.is_loaded

        await ltx2_model.unload()
        assert not ltx2_model.is_loaded

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            assert allocated < 1024**3  # Less than 1GB after unload

    @pytest.mark.asyncio
    async def test_health_check_loaded(self, ltx2_model):
        await ltx2_model.load()
        health = await ltx2_model.health_check()
        assert health["status"] == "ready"
        assert health["loaded"] is True


class TestSVDGPU:
    @pytest.mark.asyncio
    async def test_load_and_generate(self, svd_model, test_image):
        await svd_model.load()
        assert svd_model.is_loaded

        req = GenerationRequest(
            prompt="",
            image=test_image,
            num_inference_steps=5,
            seed=42,
        )
        result = await svd_model.generate(req)

        assert result.video_path.exists()
        assert result.video_path.suffix == ".mp4"
        assert result.model_name == "SVD-XT"
        assert result.mode == ModelMode.IMAGE_TO_VIDEO


class TestWanGPU:
    @pytest.fixture
    def wan_model(self):
        model = WanModel()
        yield model
        asyncio.get_event_loop().run_until_complete(model.unload())

    @pytest.mark.asyncio
    async def test_load_and_generate(self, wan_model):
        await wan_model.load()
        assert wan_model.is_loaded

        req = GenerationRequest(
            prompt="A serene mountain landscape at sunset",
            width=512,
            height=320,
            num_frames=9,
            fps=16,
            guidance_scale=5.0,
            num_inference_steps=5,
            seed=42,
        )
        result = await wan_model.generate(req)

        assert result.video_path.exists()
        assert result.video_path.suffix == ".mp4"
        assert result.model_name == "Wan 2.1"
        assert result.mode == ModelMode.TEXT_TO_VIDEO
        assert result.duration_seconds > 0


class TestModelSwapGPU:
    @pytest.mark.asyncio
    async def test_swap_ltx2_to_svd(self, ltx2_model, svd_model, test_image):
        import torch

        # Load and use LTX-2
        await ltx2_model.load()
        assert ltx2_model.is_loaded

        # Unload LTX-2
        await ltx2_model.unload()
        assert not ltx2_model.is_loaded

        # Load and use SVD-XT
        await svd_model.load()
        assert svd_model.is_loaded

        if torch.cuda.is_available():
            # SVD should use much less memory than LTX-2
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            assert allocated_gb < 10  # SVD uses ~4GB
