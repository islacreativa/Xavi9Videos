"""Stable Video Diffusion XT (SVD-XT) local inference."""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from pathlib import Path

import torch
from PIL import Image

from app.config import settings
from app.models import (
    BaseVideoModel,
    GenerationRequest,
    GenerationResult,
    ModelMode,
)

logger = logging.getLogger(__name__)

# SVD-XT fixed parameters
SVD_WIDTH = 1024
SVD_HEIGHT = 576
SVD_NUM_FRAMES = 25
SVD_FPS = 7


class SVDModel(BaseVideoModel):
    """Stable Video Diffusion XT - image-to-video only."""

    def __init__(self) -> None:
        self._pipeline = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "SVD-XT"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.IMAGE_TO_VIDEO]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_local(self) -> bool:
        return True

    async def load(self) -> None:
        if self._loaded:
            return

        logger.info("Loading SVD-XT pipeline...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)
        self._loaded = True
        logger.info("SVD-XT pipeline loaded.")

    def _load_sync(self) -> None:
        from diffusers import StableVideoDiffusionPipeline

        model_path = settings.svd_local_path
        if model_path.exists() and (model_path / "model_index.json").exists():
            source = str(model_path)
        else:
            source = settings.svd_model_id

        self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
            source,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

    async def unload(self) -> None:
        if not self._loaded:
            return

        logger.info("Unloading SVD-XT...")
        self._pipeline = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("SVD-XT unloaded, GPU memory freed.")

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.image is None:
            raise ValueError("SVD-XT requires an input image.")
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("SVD-XT model not loaded. Call load() first.")

        loop = asyncio.get_event_loop()
        start = time.time()
        output_path = await loop.run_in_executor(None, self._generate_sync, request)
        elapsed = time.time() - start

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=ModelMode.IMAGE_TO_VIDEO,
            duration_seconds=elapsed,
            metadata={
                "width": SVD_WIDTH,
                "height": SVD_HEIGHT,
                "num_frames": SVD_NUM_FRAMES,
                "fps": SVD_FPS,
                "num_inference_steps": request.num_inference_steps,
                "seed": request.seed,
            },
        )

    def _generate_sync(self, request: GenerationRequest) -> Path:
        image = request.image.resize((SVD_WIDTH, SVD_HEIGHT))

        generator = None
        if request.seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)

        result = self._pipeline(
            image=image,
            num_frames=SVD_NUM_FRAMES,
            num_inference_steps=request.num_inference_steps,
            decode_chunk_size=8,
            generator=generator,
        )
        frames = result.frames[0]

        output_path = settings.outputs_dir / f"svd_{int(time.time())}.mp4"
        self._save_frames_to_video(frames, output_path, SVD_FPS)

        return output_path

    @staticmethod
    def _save_frames_to_video(frames: list[Image.Image], path: Path, fps: int) -> None:
        import imageio
        import numpy as np

        writer = imageio.get_writer(str(path), fps=fps, codec="libx264")
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()

    async def health_check(self) -> dict:
        if self._loaded:
            status = "ready"
        else:
            model_path = settings.svd_local_path
            has_weights = model_path.exists() and any(model_path.iterdir())
            status = "available" if has_weights else "weights_missing"

        return {
            "status": status,
            "model": self.name,
            "loaded": self._loaded,
            "resolution": f"{SVD_WIDTH}x{SVD_HEIGHT}",
            "frames": SVD_NUM_FRAMES,
        }
