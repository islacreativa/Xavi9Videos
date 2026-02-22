"""Xavi9Videos - Main entry point and orchestrator."""

from __future__ import annotations

import json
import logging
import os
import traceback

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

from PIL import Image  # noqa: E402

from app.config import settings  # noqa: E402
from app.models import (  # noqa: E402
    BaseVideoModel,
    GenerationRequest,
    ModelMode,
    generation_lock,
)
from app.models.cosmos import CosmosText2World, CosmosVideo2World  # noqa: E402
from app.models.ltx2 import LTX2Model  # noqa: E402
from app.models.nvidia_build import (  # noqa: E402
    NvidiaBuildText2World,
    NvidiaBuildVideo2World,
)
from app.models.svd import SVDModel  # noqa: E402
from app.models.wan import WanModel  # noqa: E402
from app.ui.components import build_ui  # noqa: E402
from app.utils.storage import cleanup_old_outputs  # noqa: E402


def _setup_logging() -> None:
    """Configure logging: JSON format in production, human-readable in dev."""
    log_format = os.environ.get("LOG_FORMAT", "text")
    level = logging.INFO

    if log_format == "json":

        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info and record.exc_info[0]:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_entry, ensure_ascii=False)

        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logging.root.handlers = [handler]
        logging.root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


_setup_logging()
logger = logging.getLogger(__name__)

# --- Model Registry ---

models: dict[str, BaseVideoModel] = {}

# Cosmos requires a running NIM container (no ARM64 image yet for DGX Spark)
if settings.cosmos_nim_url:
    models["Cosmos Text2World"] = CosmosText2World()
    models["Cosmos Video2World"] = CosmosVideo2World()
    logger.info("Cosmos models enabled (NIM URL: %s)", settings.cosmos_nim_url)
else:
    logger.info("Cosmos models disabled (COSMOS_NIM_URL not set)")

# NVIDIA Build API cloud models (requires API key)
if settings.nvidia_api_key:
    models["Cloud: Cosmos Text2World"] = NvidiaBuildText2World()
    models["Cloud: Cosmos Video2World"] = NvidiaBuildVideo2World()
    logger.info("NVIDIA Build API models enabled (cloud)")
else:
    logger.info("NVIDIA Build API models disabled (NVIDIA_API_KEY not set)")

# Local models always available
models["LTX-2"] = LTX2Model()
models["SVD-XT"] = SVDModel()
models["Wan 2.1"] = WanModel()

# Track which local model is currently loaded
_current_local_model: str | None = None


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
    progress_callback=None,
) -> tuple[str | None, dict | None, str]:
    """Main generation function called by Gradio UI."""

    if model_name not in models:
        return None, None, f"Unknown model: {model_name}"

    model = models[model_name]

    # Validate inputs
    if (
        ModelMode.IMAGE_TO_VIDEO in model.supported_modes
        and ModelMode.TEXT_TO_VIDEO not in model.supported_modes
        and image is None
    ):
        return None, None, f"{model_name} requires an input image."

    if ModelMode.TEXT_TO_VIDEO in model.supported_modes and not image and not prompt.strip():
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
            try:
                result = await model.generate(request, progress_callback=progress_callback)
            except TypeError:
                result = await model.generate(request)

        cleanup_old_outputs()

        metadata = {
            "model": result.model_name,
            "mode": result.mode.value,
            "generation_time": f"{result.duration_seconds:.1f}s",
            **result.metadata,
        }

        status = f"Generated with {result.model_name} in {result.duration_seconds:.1f}s"

        return str(result.video_path), metadata, status

    except ValueError as e:
        return None, None, f"Validation error: {e}"
    except TimeoutError:
        return (
            None,
            None,
            ("Generation timed out. Try a shorter video (fewer frames) or fewer inference steps."),
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            # Try to recover
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None, ("Out of GPU memory. Try reducing resolution or frame count.")
        if "model not found" in error_msg.lower() or "no such file" in error_msg.lower():
            return None, None, ("Model weights not found. Run: ./scripts/download_models.sh")
        return None, None, f"Runtime error: {e}"
    except Exception as e:
        logger.error("Generation failed: %s\n%s", e, traceback.format_exc())
        if "cosmos" in model_name.lower() and (
            "connect" in str(e).lower() or "refused" in str(e).lower()
        ):
            return (
                None,
                None,
                (
                    "Cannot reach Cosmos NIM. Check that it's running: "
                    "docker compose restart cosmos-nim"
                ),
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
