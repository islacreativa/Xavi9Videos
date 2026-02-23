"""Shared base class for fal.ai cloud video generation models."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import httpx

from app.config import settings
from app.models import BaseVideoModel

logger = logging.getLogger(__name__)

_QUEUE_BASE = "https://queue.fal.run"
_POLL_INTERVAL = 5.0
_MAX_POLLS = 120


class FalBaseModel(BaseVideoModel):
    """Base class for fal.ai cloud models.

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
            resp = await client.get(self._endpoint)
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
