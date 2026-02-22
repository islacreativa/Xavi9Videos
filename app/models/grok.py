"""Grok Imagine (xAI) cloud video generation model."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time

import httpx

from app.config import settings
from app.models import (
    BaseVideoModel,
    GenerationRequest,
    GenerationResult,
    ModelMode,
)

logger = logging.getLogger(__name__)

_API_BASE = "https://api.x.ai"
_POLL_INTERVAL = 5.0
_MAX_POLLS = 120


class GrokVideoModel(BaseVideoModel):
    """Grok Imagine video generation via xAI API.

    Supports both text-to-video and image-to-video modes.
    Uses async polling: POST to submit, GET to poll status, download MP4 URL.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=_API_BASE,
                headers={
                    "Authorization": f"Bearer {settings.grok_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(settings.grok_api_timeout),
            )
        return self._client

    @property
    def name(self) -> str:
        return "Cloud: Grok Video"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.TEXT_TO_VIDEO, ModelMode.IMAGE_TO_VIDEO]

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_loaded(self) -> bool:
        return True

    async def generate(
        self, request: GenerationRequest, progress_callback=None
    ) -> GenerationResult:
        if not request.prompt and request.image is None:
            raise ValueError("Grok Video requires a text prompt or an input image.")

        payload: dict = {
            "model": "grok-2-image",
            "prompt": request.prompt or "Generate a video from this image",
        }

        # Image-to-video: encode image as base64 data URI
        if request.image is not None:
            buf = io.BytesIO()
            request.image.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode()
            payload["image_url"] = f"data:image/png;base64,{image_b64}"

        start = time.time()
        client = self._get_client()

        # Submit generation request
        response = await client.post("/v1/videos/generations", json=payload)
        response.raise_for_status()
        data = response.json()

        request_id = data.get("request_id")
        if not request_id:
            raise RuntimeError("Grok API did not return a request_id")

        logger.info("Grok Video: submitted request %s, polling...", request_id)

        # Poll until done
        video_url = await self._poll_result(request_id)
        elapsed = time.time() - start

        # Download video
        video_resp = await client.get(video_url)
        video_resp.raise_for_status()

        output_path = settings.outputs_dir / f"grok_{int(time.time())}.mp4"
        output_path.write_bytes(video_resp.content)

        mode = ModelMode.IMAGE_TO_VIDEO if request.image else ModelMode.TEXT_TO_VIDEO
        metadata: dict = {"prompt": request.prompt or ""}
        if request.image:
            metadata["mode"] = "image-to-video"

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=mode,
            duration_seconds=elapsed,
            metadata=metadata,
        )

    async def _poll_result(self, request_id: str) -> str:
        """Poll Grok API until video is ready. Returns the video download URL."""
        client = self._get_client()

        for _ in range(_MAX_POLLS):
            await asyncio.sleep(_POLL_INTERVAL)
            resp = await client.get(f"/v1/videos/{request_id}")
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status", "")
            if status == "done":
                video_info = data.get("video", {})
                url = video_info.get("url")
                if not url:
                    raise RuntimeError("Grok API returned done but no video URL")
                return url
            if status in ("failed", "error"):
                error_msg = data.get("error", "Unknown error")
                raise RuntimeError(f"Grok video generation failed: {error_msg}")

        raise TimeoutError(f"Grok API polling timed out after {_MAX_POLLS * _POLL_INTERVAL:.0f}s")

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            resp = await client.get("/v1/models")
            if resp.status_code == 200:
                return {"status": "ready", "model": self.name, "type": "cloud"}
            return {
                "status": "available",
                "model": self.name,
                "type": "cloud",
                "note": "API key configured",
            }
        except Exception as e:
            return {"status": "error", "model": self.name, "type": "cloud", "error": str(e)}
