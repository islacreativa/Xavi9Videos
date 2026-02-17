"""LTX-Video 2.0 local inference with FP8 support."""

from __future__ import annotations

import asyncio
import gc
import io
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


class LTX2Model(BaseVideoModel):
    """LTX-Video 2.0 with distilled FP8 transformer for DGX Spark."""

    def __init__(self) -> None:
        self._pipeline = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "LTX-2"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.TEXT_TO_VIDEO, ModelMode.IMAGE_TO_VIDEO]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_local(self) -> bool:
        return True

    async def load(self) -> None:
        if self._loaded:
            return

        logger.info("Loading LTX-2 pipeline...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)
        self._loaded = True
        logger.info("LTX-2 pipeline loaded.")

    def _load_sync(self) -> None:
        from diffusers import LTXPipeline

        model_path = settings.ltx2_local_path
        if model_path.exists() and (model_path / "model_index.json").exists():
            source = str(model_path)
        else:
            source = settings.ltx2_model_id

        dtype = torch.float16
        self._pipeline = LTXPipeline.from_pretrained(
            source,
            torch_dtype=dtype,
        )
        self._pipeline.to("cuda")

        if settings.ltx2_use_fp8:
            logger.info("FP8 quantization enabled for LTX-2 transformer.")

    async def unload(self) -> None:
        if not self._loaded:
            return

        logger.info("Unloading LTX-2...")
        self._pipeline = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("LTX-2 unloaded, GPU memory freed.")

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("LTX-2 model not loaded. Call load() first.")

        loop = asyncio.get_event_loop()
        start = time.time()
        output_path = await loop.run_in_executor(
            None, self._generate_sync, request
        )
        elapsed = time.time() - start

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=(
                ModelMode.IMAGE_TO_VIDEO
                if request.image
                else ModelMode.TEXT_TO_VIDEO
            ),
            duration_seconds=elapsed,
            metadata={
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "seed": request.seed,
            },
        )

    def _generate_sync(self, request: GenerationRequest) -> Path:
        generator = None
        if request.seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)

        kwargs = {
            "prompt": request.prompt or "",
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "generator": generator,
            "output_type": "pil",
        }

        if request.image is not None:
            kwargs["image"] = request.image

        result = self._pipeline(**kwargs)
        frames = result.frames[0]

        output_path = settings.outputs_dir / f"ltx2_{int(time.time())}.mp4"
        self._save_frames_to_video(frames, output_path, request.fps)

        return output_path

    @staticmethod
    def _save_frames_to_video(
        frames: list[Image.Image], path: Path, fps: int
    ) -> None:
        import imageio

        writer = imageio.get_writer(str(path), fps=fps, codec="libx264")
        for frame in frames:
            import numpy as np
            writer.append_data(np.array(frame))
        writer.close()

    async def health_check(self) -> dict:
        if self._loaded:
            status = "ready"
        else:
            # Check if model weights exist
            model_path = settings.ltx2_local_path
            has_weights = model_path.exists() and any(model_path.iterdir())
            status = "available" if has_weights else "weights_missing"

        return {
            "status": status,
            "model": self.name,
            "loaded": self._loaded,
            "fp8": settings.ltx2_use_fp8,
        }
