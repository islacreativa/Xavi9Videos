"""NVIDIA Build API (build.nvidia.com) cloud video generation models."""

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

_POLL_INTERVAL = 5.0  # seconds between polling attempts
_MAX_POLLS = 120  # max polling attempts (~10 min at 5s interval)


class NvidiaBuildModel(BaseVideoModel):
    """Base class for NVIDIA Build API cloud models.

    Handles Bearer token auth, async polling for HTTP 202 responses,
    and shared httpx client lifecycle.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.nvidia_api_url,
                headers={
                    "Authorization": f"Bearer {settings.nvidia_api_key}",
                    "Accept": "application/json",
                },
                timeout=httpx.Timeout(settings.nvidia_api_timeout),
            )
        return self._client

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_loaded(self) -> bool:
        return True

    async def _poll_result(self, request_id: str) -> dict:
        """Poll NVCF for an async result until completion."""
        client = self._get_client()
        poll_url = f"/v1/status/{request_id}"

        for _ in range(_MAX_POLLS):
            await asyncio.sleep(_POLL_INTERVAL)
            resp = await client.get(poll_url)

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 202:
                continue  # still processing

            resp.raise_for_status()

        raise TimeoutError(
            f"NVIDIA Build API polling timed out after {_MAX_POLLS * _POLL_INTERVAL:.0f}s"
        )

    async def _call_api(self, payload: dict) -> dict:
        """Send request to the API, handling async polling if needed."""
        client = self._get_client()
        response = await client.post(self._endpoint, json=payload)

        if response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID", "")
            if not request_id:
                raise RuntimeError("Got HTTP 202 but no NVCF-REQID header for polling")
            logger.info("NVIDIA Build API: async request %s, polling...", request_id)
            return await self._poll_result(request_id)

        response.raise_for_status()
        return response.json()

    async def health_check(self) -> dict:
        """Check API connectivity with a lightweight request."""
        try:
            client = self._get_client()
            resp = await client.get("/v1/health/ready")
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


class NvidiaBuildText2World(NvidiaBuildModel):
    """Cosmos 1.0 7B Text2World via NVIDIA Build API."""

    def __init__(self) -> None:
        super().__init__("/v1/cosmos/nvidia/cosmos-1.0-7b-diffusion-text2world")

    @property
    def name(self) -> str:
        return "Cloud: Cosmos Text2World"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.TEXT_TO_VIDEO]

    async def generate(
        self, request: GenerationRequest, progress_callback=None
    ) -> GenerationResult:
        if not request.prompt:
            raise ValueError("Text2World requires a text prompt.")

        payload: dict = {
            "prompt": request.prompt,
            "video_params": {
                "frames_count": request.num_frames,
                "frames_per_sec": request.fps,
            },
        }
        if request.seed >= 0:
            payload["seed"] = request.seed

        start = time.time()
        data = await self._call_api(payload)
        elapsed = time.time() - start

        video_bytes = base64.b64decode(data["b64_video"])
        output_path = settings.outputs_dir / f"nvbuild_t2w_{int(time.time())}.mp4"
        output_path.write_bytes(video_bytes)

        metadata: dict = {"prompt": request.prompt}
        if "seed" in data:
            metadata["seed"] = data["seed"]
        if "upsampled_prompt" in data:
            metadata["upsampled_prompt"] = data["upsampled_prompt"]

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=ModelMode.TEXT_TO_VIDEO,
            duration_seconds=elapsed,
            metadata=metadata,
        )


class NvidiaBuildVideo2World(NvidiaBuildModel):
    """Cosmos 1.0 7B Video2World (image-conditioned) via NVIDIA Build API."""

    def __init__(self) -> None:
        super().__init__("/v1/cosmos/nvidia/cosmos-1.0-7b-diffusion-video2world")

    @property
    def name(self) -> str:
        return "Cloud: Cosmos Video2World"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.IMAGE_TO_VIDEO]

    async def generate(
        self, request: GenerationRequest, progress_callback=None
    ) -> GenerationResult:
        if request.image is None:
            raise ValueError("Video2World requires an input image.")

        buf = io.BytesIO()
        request.image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        payload: dict = {
            "image": image_b64,
            "video_params": {
                "frames_count": request.num_frames,
                "frames_per_sec": request.fps,
            },
        }
        if request.prompt:
            payload["prompt"] = request.prompt
        if request.seed >= 0:
            payload["seed"] = request.seed

        start = time.time()
        data = await self._call_api(payload)
        elapsed = time.time() - start

        video_bytes = base64.b64decode(data["b64_video"])
        output_path = settings.outputs_dir / f"nvbuild_v2w_{int(time.time())}.mp4"
        output_path.write_bytes(video_bytes)

        metadata: dict = {"prompt": request.prompt or ""}
        if "seed" in data:
            metadata["seed"] = data["seed"]
        if "upsampled_prompt" in data:
            metadata["upsampled_prompt"] = data["upsampled_prompt"]

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=ModelMode.IMAGE_TO_VIDEO,
            duration_seconds=elapsed,
            metadata=metadata,
        )
