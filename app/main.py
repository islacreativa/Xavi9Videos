"""Xavi9Videos - Main entry point and orchestrator."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Dict, Optional, Tuple

# Patch gradio_client bug with additionalProperties: true on Python 3.9
import gradio_client.utils as _gc_utils
_orig_json_schema = _gc_utils._json_schema_to_python_type
def _patched_json_schema(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    if schema == {} or schema is None:
        return "Any"
    return _orig_json_schema(schema, defs)
_gc_utils._json_schema_to_python_type = _patched_json_schema

import gradio as gr
from PIL import Image

from app.config import settings
from app.models import (
    BaseVideoModel,
    GenerationRequest,
    ModelMode,
    generation_lock,
)
from app.models.cosmos import CosmosText2World, CosmosVideo2World
from app.models.ltx2 import LTX2Model
from app.models.svd import SVDModel
from app.ui.components import build_ui
from app.utils.storage import cleanup_old_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Model Registry ---

models: Dict[str, BaseVideoModel] = {
    "Cosmos Text2World": CosmosText2World(),
    "Cosmos Video2World": CosmosVideo2World(),
    "LTX-2": LTX2Model(),
    "SVD-XT": SVDModel(),
}

# Track which local model is currently loaded
_current_local_model: Optional[str] = None


async def _ensure_model_loaded(model_name: str) -> None:
    """Load the requested model, unloading the current local model if needed."""
    global _current_local_model

    model = models[model_name]
    if not model.is_local:
        return  # Cosmos runs in its own container

    if model.is_loaded:
        return

    # Unload current local model if different
    if _current_local_model and _current_local_model != model_name:
        current = models[_current_local_model]
        logger.info("Swapping model: %s -> %s", _current_local_model, model_name)
        await current.unload()

    await model.load()
    _current_local_model = model_name


async def generate_video(
    model_name: str,
    prompt: str,
    image: Image.Image | None,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[str | None, dict | None, str]:
    """Main generation function called by Gradio UI."""

    if model_name not in models:
        return None, None, f"Unknown model: {model_name}"

    model = models[model_name]

    # Validate inputs
    if ModelMode.IMAGE_TO_VIDEO in model.supported_modes and ModelMode.TEXT_TO_VIDEO not in model.supported_modes:
        if image is None:
            return None, None, f"{model_name} requires an input image."

    if ModelMode.TEXT_TO_VIDEO in model.supported_modes and not image:
        if not prompt.strip():
            return None, None, "Please provide a text prompt."

    request = GenerationRequest(
        prompt=prompt or "",
        image=image,
        width=int(width),
        height=int(height),
        num_frames=int(num_frames),
        fps=int(fps),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        seed=int(seed),
    )

    try:
        async with generation_lock:
            await _ensure_model_loaded(model_name)
            result = await model.generate(request)

        cleanup_old_outputs()

        metadata = {
            "model": result.model_name,
            "mode": result.mode.value,
            "generation_time": f"{result.duration_seconds:.1f}s",
            **result.metadata,
        }

        status = (
            f"Generated with {result.model_name} in "
            f"{result.duration_seconds:.1f}s"
        )

        return str(result.video_path), metadata, status

    except ValueError as e:
        return None, None, f"Validation error: {e}"
    except TimeoutError:
        return None, None, (
            "Generation timed out. Try a shorter video "
            "(fewer frames) or fewer inference steps."
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            # Try to recover
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None, (
                "Out of GPU memory. Try reducing resolution or frame count."
            )
        if "model not found" in error_msg.lower() or "no such file" in error_msg.lower():
            return None, None, (
                f"Model weights not found. Run: ./scripts/download_models.sh"
            )
        return None, None, f"Runtime error: {e}"
    except Exception as e:
        logger.error("Generation failed: %s\n%s", e, traceback.format_exc())
        if "cosmos" in model_name.lower() and ("connect" in str(e).lower() or "refused" in str(e).lower()):
            return None, None, (
                "Cannot reach Cosmos NIM. Check that it's running: "
                "docker compose restart cosmos-nim"
            )
        return None, None, f"Error: {e}"


async def check_health() -> dict:
    """Check health of all models."""
    results = {}
    for name, model in models.items():
        results[name] = await model.health_check()
    return results


def main() -> None:
    """Launch the Gradio application."""
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Xavi9Videos...")
    logger.info("Cosmos NIM URL: %s", settings.cosmos_nim_url)
    logger.info("Models dir: %s", settings.models_dir)
    logger.info("Outputs dir: %s", settings.outputs_dir)

    demo = build_ui(
        generate_fn=generate_video,
        health_check_fn=check_health,
    )

    demo.queue(
        max_size=5,
        default_concurrency_limit=settings.max_concurrent_requests,
    )

    import os
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    demo.launch(
        server_name=host,
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()
