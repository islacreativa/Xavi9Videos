"""fal.ai Kling v3 Pro cloud video generation models."""

from __future__ import annotations

import base64
import io
import logging
import time

from app.models import (
    GenerationRequest,
    GenerationResult,
    ModelMode,
)
from app.models.fal_base import FalBaseModel

logger = logging.getLogger(__name__)


class FalKlingTextToVideo(FalBaseModel):
    """Kling v3 Pro text-to-video via fal.ai."""

    def __init__(self) -> None:
        super().__init__("/fal-ai/kling-video/v3/pro/text-to-video")

    @property
    def name(self) -> str:
        return "Cloud: Kling v3"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.TEXT_TO_VIDEO]

    async def generate(
        self, request: GenerationRequest, progress_callback=None
    ) -> GenerationResult:
        if not request.prompt:
            raise ValueError("Kling v3 Pro text-to-video requires a text prompt.")

        payload: dict = {
            "prompt": request.prompt,
            "duration": "5",
            "aspect_ratio": "16:9",
        }
        if request.seed >= 0:
            payload["seed"] = request.seed

        start = time.time()
        request_id = await self._submit(payload)
        logger.info("fal.ai Kling T2V: submitted %s, polling...", request_id)

        result_data = await self._poll_result(request_id)
        elapsed = time.time() - start

        video_info = result_data.get("video", {})
        video_url = video_info.get("url")
        if not video_url:
            raise RuntimeError("fal.ai response missing video URL")

        output_path = await self._download_video(video_url, "fal_kling_t2v")

        metadata: dict = {"prompt": request.prompt}
        if "duration" in video_info:
            metadata["video_duration"] = video_info["duration"]

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=ModelMode.TEXT_TO_VIDEO,
            duration_seconds=elapsed,
            metadata=metadata,
        )


class FalKlingImageToVideo(FalBaseModel):
    """Kling v3 Pro image-to-video via fal.ai."""

    def __init__(self) -> None:
        super().__init__("/fal-ai/kling-video/v3/pro/image-to-video")

    @property
    def name(self) -> str:
        return "Cloud: Kling v3 I2V"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.IMAGE_TO_VIDEO]

    async def generate(
        self, request: GenerationRequest, progress_callback=None
    ) -> GenerationResult:
        if request.image is None:
            raise ValueError("Kling v3 Pro image-to-video requires an input image.")

        buf = io.BytesIO()
        request.image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        payload: dict = {
            "prompt": request.prompt or "",
            "start_image_url": f"data:image/png;base64,{image_b64}",
            "duration": "5",
            "aspect_ratio": "16:9",
        }
        if request.seed >= 0:
            payload["seed"] = request.seed

        start = time.time()
        request_id = await self._submit(payload)
        logger.info("fal.ai Kling I2V: submitted %s, polling...", request_id)

        result_data = await self._poll_result(request_id)
        elapsed = time.time() - start

        video_info = result_data.get("video", {})
        video_url = video_info.get("url")
        if not video_url:
            raise RuntimeError("fal.ai response missing video URL")

        output_path = await self._download_video(video_url, "fal_kling_i2v")

        metadata: dict = {"prompt": request.prompt or ""}
        if "duration" in video_info:
            metadata["video_duration"] = video_info["duration"]

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=ModelMode.IMAGE_TO_VIDEO,
            duration_seconds=elapsed,
            metadata=metadata,
        )
