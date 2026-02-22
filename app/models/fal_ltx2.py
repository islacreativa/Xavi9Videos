"""fal.ai LTX-2 Pro cloud video generation models."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from pathlib import Path

import httpx

from app.config import settings
from app.models import (
    BaseVideoModel,
    GenerationRequest,
    GenerationResult,
    ModelMode,
)

logger = logging.getLogger(__name__)

_QUEUE_BASE = "https://queue.fal.run"
_POLL_INTERVAL = 5.0
_MAX_POLLS = 120


class _FalLTX2Base(BaseVideoModel):
    """Base class for fal.ai LTX-2 Pro models.

    Handles auth, submit/poll/result pattern, and video download.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=_QUEUE_BASE,
                headers={
                    "Authorization": f"Key {settings.fal_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(settings.fal_api_timeout),
            )
        return self._client

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_loaded(self) -> bool:
        return True

    async def _submit(self, payload: dict) -> str:
        """Submit a generation request. Returns request_id."""
        client = self._get_client()
        resp = await client.post(self._endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
        request_id = data.get("request_id")
        if not request_id:
            raise RuntimeError("fal.ai did not return a request_id")
        return request_id

    async def _poll_result(self, request_id: str) -> dict:
        """Poll fal.ai until the request completes. Returns the result dict."""
        client = self._get_client()
        status_url = f"{self._endpoint}/requests/{request_id}/status"
        result_url = f"{self._endpoint}/requests/{request_id}"

        for _ in range(_MAX_POLLS):
            await asyncio.sleep(_POLL_INTERVAL)
            resp = await client.get(status_url)
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status", "")
            if status == "COMPLETED":
                # Fetch full result
                result_resp = await client.get(result_url)
                result_resp.raise_for_status()
                return result_resp.json()
            if status in ("FAILED", "ERROR"):
                error_msg = data.get("error", "Unknown error")
                raise RuntimeError(f"fal.ai generation failed: {error_msg}")

        raise TimeoutError(f"fal.ai polling timed out after {_MAX_POLLS * _POLL_INTERVAL:.0f}s")

    async def _download_video(self, video_url: str, prefix: str) -> Path:
        """Download MP4 from URL and save to outputs dir. Returns file path."""
        client = self._get_client()
        resp = await client.get(video_url)
        resp.raise_for_status()

        output_path = settings.outputs_dir / f"{prefix}_{int(time.time())}.mp4"
        output_path.write_bytes(resp.content)
        return output_path

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            # Light probe: check that we can reach the queue endpoint
            resp = await client.get(self._endpoint)
            # fal.ai returns 200 or 405 for GET on a POST endpoint
            if resp.status_code in (200, 405, 422):
                return {"status": "ready", "model": self.name, "type": "cloud"}
            return {
                "status": "available",
                "model": self.name,
                "type": "cloud",
                "note": "API key configured",
            }
        except Exception as e:
            return {"status": "error", "model": self.name, "type": "cloud", "error": str(e)}


class FalLTX2TextToVideo(_FalLTX2Base):
    """LTX-2 Pro text-to-video via fal.ai."""

    def __init__(self) -> None:
        super().__init__("/fal-ai/ltx-2/text-to-video")

    @property
    def name(self) -> str:
        return "Cloud: LTX-2 Pro"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.TEXT_TO_VIDEO]

    async def generate(
        self, request: GenerationRequest, progress_callback=None
    ) -> GenerationResult:
        if not request.prompt:
            raise ValueError("LTX-2 Pro text-to-video requires a text prompt.")

        payload: dict = {
            "prompt": request.prompt,
            "num_frames": request.num_frames,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
        }
        if request.seed >= 0:
            payload["seed"] = request.seed

        start = time.time()
        request_id = await self._submit(payload)
        logger.info("fal.ai LTX-2 T2V: submitted %s, polling...", request_id)

        result_data = await self._poll_result(request_id)
        elapsed = time.time() - start

        video_info = result_data.get("video", {})
        video_url = video_info.get("url")
        if not video_url:
            raise RuntimeError("fal.ai response missing video URL")

        output_path = await self._download_video(video_url, "fal_ltx2_t2v")

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


class FalLTX2ImageToVideo(_FalLTX2Base):
    """LTX-2 Pro image-to-video via fal.ai."""

    def __init__(self) -> None:
        super().__init__("/fal-ai/ltx-2/image-to-video")

    @property
    def name(self) -> str:
        return "Cloud: LTX-2 Pro I2V"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.IMAGE_TO_VIDEO]

    async def generate(
        self, request: GenerationRequest, progress_callback=None
    ) -> GenerationResult:
        if request.image is None:
            raise ValueError("LTX-2 Pro image-to-video requires an input image.")

        # Encode image as base64 data URI
        buf = io.BytesIO()
        request.image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        payload: dict = {
            "prompt": request.prompt or "",
            "image_url": f"data:image/png;base64,{image_b64}",
            "num_frames": request.num_frames,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
        }
        if request.seed >= 0:
            payload["seed"] = request.seed

        start = time.time()
        request_id = await self._submit(payload)
        logger.info("fal.ai LTX-2 I2V: submitted %s, polling...", request_id)

        result_data = await self._poll_result(request_id)
        elapsed = time.time() - start

        video_info = result_data.get("video", {})
        video_url = video_info.get("url")
        if not video_url:
            raise RuntimeError("fal.ai response missing video URL")

        output_path = await self._download_video(video_url, "fal_ltx2_i2v")

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
