"""Video model abstractions, dataclasses, and shared lock.

Adding a new model
==================
1. Create ``app/models/<name>.py`` with a class that inherits ``BaseVideoModel``.
2. Implement the four abstract members:
   - ``name`` (property) - human-readable name shown in the UI.
   - ``supported_modes`` (property) - list of ``ModelMode`` values.
   - ``generate(request)`` - produce a video and return ``GenerationResult``.
   - ``health_check()`` - return a dict with at least ``{"status": ..., "model": ...}``.
3. For **local** models (running on the GPU), also override:
   - ``is_local`` → ``True``
   - ``is_loaded`` - whether the pipeline is in memory.
   - ``load()`` / ``unload()`` - manage GPU memory.
4. Register the model in ``app/main.py`` by adding it to the ``models`` dict.
5. Add the model name to ``MODEL_CHOICES`` in ``app/ui/components.py``.
6. (Optional) Add config fields in ``app/config.py`` (model ID, local path, etc.).

See ``app/models/ltx2.py`` (local, text+image) and ``app/models/cosmos.py``
(remote NIM API) as reference implementations.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PIL import Image


class ModelMode(str, Enum):
    TEXT_TO_VIDEO = "text2video"
    IMAGE_TO_VIDEO = "image2video"


@dataclass
class GenerationRequest:
    """Parameters for a video generation request."""

    prompt: str = ""
    image: Image.Image | None = None
    width: int = 768
    height: int = 512
    num_frames: int = 49
    fps: int = 24
    guidance_scale: float = 3.0
    num_inference_steps: int = 30
    seed: int = -1


@dataclass
class GenerationResult:
    """Result of a video generation."""

    video_path: Path
    model_name: str
    mode: ModelMode
    duration_seconds: float
    metadata: dict = field(default_factory=dict)


class BaseVideoModel(ABC):
    """Abstract base for all video generation models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @property
    @abstractmethod
    def supported_modes(self) -> list[ModelMode]:
        """List of supported generation modes."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a video from the request."""

    @abstractmethod
    async def health_check(self) -> dict:
        """Return health status of the model."""

    async def load(self) -> None:  # noqa: B027
        """Load model into GPU memory. Override for local models."""

    async def unload(self) -> None:  # noqa: B027
        """Unload model from GPU memory. Override for local models."""

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded. Override for local models."""
        return True

    @property
    def is_local(self) -> bool:
        """Whether this model runs locally (needs GPU memory management)."""
        return False


# Shared lock for serializing generation requests (1 at a time)
generation_lock = asyncio.Lock()
