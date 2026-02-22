"""Cosmos Predict1 7B client via NVIDIA NIM REST API."""

from __future__ import annotations

import base64
import io
import time

import httpx

from app.config import settings
from app.models import (
    BaseVideoModel,
    GenerationRequest,
    GenerationResult,
    ModelMode,
)


class CosmosText2World(BaseVideoModel):
    """Cosmos Predict1 Text2World via NIM API."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "Cosmos Text2World"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.TEXT_TO_VIDEO]

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.cosmos_nim_url,
                timeout=httpx.Timeout(settings.cosmos_timeout_seconds),
            )
        return self._client

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if not request.prompt:
            raise ValueError("Cosmos Text2World requires a text prompt.")

        client = self._get_client()
        payload = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "fps": request.fps,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
        }
        if request.seed >= 0:
            payload["seed"] = request.seed

        start = time.time()
        response = await client.post("/v1/infer", json=payload)
        response.raise_for_status()
        elapsed = time.time() - start

        data = response.json()
        video_bytes = base64.b64decode(data["video"])

        output_path = settings.outputs_dir / f"cosmos_t2w_{int(time.time())}.mp4"
        output_path.write_bytes(video_bytes)

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=ModelMode.TEXT_TO_VIDEO,
            duration_seconds=elapsed,
            metadata={"prompt": request.prompt, **payload},
        )

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            resp = await client.get("/v1/health/ready")
            ready = resp.status_code == 200
            return {"status": "ready" if ready else "not_ready", "model": self.name}
        except Exception as e:
            return {"status": "error", "model": self.name, "error": str(e)}


class CosmosVideo2World(BaseVideoModel):
    """Cosmos Predict1 Video2World (image-conditioned) via NIM API."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "Cosmos Video2World"

    @property
    def supported_modes(self) -> list[ModelMode]:
        return [ModelMode.IMAGE_TO_VIDEO]

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.cosmos_nim_url,
                timeout=httpx.Timeout(settings.cosmos_timeout_seconds),
            )
        return self._client

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.image is None:
            raise ValueError("Cosmos Video2World requires an input image.")

        client = self._get_client()

        buf = io.BytesIO()
        request.image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "image": image_b64,
            "prompt": request.prompt or "",
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "fps": request.fps,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
        }
        if request.seed >= 0:
            payload["seed"] = request.seed

        start = time.time()
        response = await client.post("/v1/infer", json=payload)
        response.raise_for_status()
        elapsed = time.time() - start

        data = response.json()
        video_bytes = base64.b64decode(data["video"])

        output_path = settings.outputs_dir / f"cosmos_v2w_{int(time.time())}.mp4"
        output_path.write_bytes(video_bytes)

        return GenerationResult(
            video_path=output_path,
            model_name=self.name,
            mode=ModelMode.IMAGE_TO_VIDEO,
            duration_seconds=elapsed,
            metadata={"prompt": request.prompt, **payload},
        )

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            resp = await client.get("/v1/health/ready")
            ready = resp.status_code == 200
            return {"status": "ready" if ready else "not_ready", "model": self.name}
        except Exception as e:
            return {"status": "error", "model": self.name, "error": str(e)}
