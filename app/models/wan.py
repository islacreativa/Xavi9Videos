"""Wan 2.1 Video local inference (text-to-video and image-to-video)."""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from pathlib import Path

import torch

from app.config import settings
from app.models import (
    BaseVideoModel,
    GenerationRequest,
    GenerationResult,
    ModelMode,
)

logger = logging.getLogger(__name__)


class WanModel(BaseVideoModel):
    """Wan 2.1 video generation (14B T2V / I2V 720p)."""

    def __init__(self) -> None:
        self._pipeline = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "Wan 2.1"

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

        logger.info("Loading Wan 2.1 pipeline...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)
        self._loaded = True
        logger.info("Wan 2.1 pipeline loaded.")

    def _load_sync(self) -> None:
        from diffusers import AutoencoderKLWan, WanPipeline

        model_id = settings.wan_model_id
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        self._pipeline = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    async def unload(self) -> None:
        if not self._loaded:
            return

        logger.info("Unloading Wan 2.1...")
        self._pipeline = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Wan 2.1 unloaded, GPU memory freed.")

    async def generate(
        self, request: GenerationRequest, progress_callback=None
    ) -> GenerationResult:
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("Wan 2.1 model not loaded. Call load() first.")

        loop = asyncio.get_event_loop()
        start = time.time()
        output_path = await loop.run_in_executor(
            None, self._generate_sync, request, progress_callback
        )
        elapsed = time.time() - start

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=(ModelMode.IMAGE_TO_VIDEO if request.image else ModelMode.TEXT_TO_VIDEO),
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

    def _generate_sync(self, request: GenerationRequest, progress_callback=None) -> Path:
        generator = None
        if request.seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)

        negative_prompt = (
            "Bright tones, overexposed, static, blurred details, "
            "subtitles, style, works, paintings, images, static, "
            "overall gray, worst quality, low quality, JPEG artifacts, "
            "ugly, incomplete, extra fingers, poorly drawn hands, "
            "poorly drawn faces, deformed, disfigured, misshapen limbs, "
            "fused fingers, still picture, messy background, "
            "three legs, many people in the background, walking backwards"
        )

        kwargs = {
            "prompt": request.prompt or "",
            "negative_prompt": negative_prompt,
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "generator": generator,
        }

        if progress_callback is not None:
            total_steps = request.num_inference_steps

            def callback_on_step(pipe, step, timestep, cb_kwargs):
                progress_callback(step + 1, total_steps)
                return cb_kwargs

            kwargs["callback_on_step_end"] = callback_on_step

        output = self._pipeline(**kwargs)
        frames = output.frames[0]

        output_path = settings.outputs_dir / f"wan_{int(time.time())}.mp4"

        from diffusers.utils import export_to_video

        export_to_video(frames, str(output_path), fps=request.fps)

        return output_path

    async def health_check(self) -> dict:
        if self._loaded:
            status = "ready"
        else:
            status = "available"

        return {
            "status": status,
            "model": self.name,
            "loaded": self._loaded,
        }
