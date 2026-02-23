"""Xavi9Videos - Main entry point and orchestrator."""

from __future__ import annotations

import json
import logging
import os
import traceback
from pathlib import Path

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

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import settings  # noqa: E402
from app.models import (  # noqa: E402
    BaseVideoModel,
    GenerationRequest,
    ModelMode,
    generation_lock,
)
from app.models.cosmos import CosmosText2World, CosmosVideo2World  # noqa: E402
from app.models.fal_kling import FalKlingImageToVideo, FalKlingTextToVideo  # noqa: E402
from app.models.fal_ltx2 import FalLTX2ImageToVideo, FalLTX2TextToVideo  # noqa: E402
from app.models.grok import GrokVideoModel  # noqa: E402
from app.models.ltx2 import LTX2Model  # noqa: E402
from app.models.svd import SVDModel  # noqa: E402
from app.models.wan import WanModel  # noqa: E402
from app.ui.components import MODEL_CHOICES, build_ui  # noqa: E402
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

# Grok Imagine cloud model (requires xAI API key)
if settings.grok_api_key:
    models["Cloud: Grok Video"] = GrokVideoModel()
    logger.info("Grok Video model enabled (cloud)")
else:
    logger.info("Grok Video model disabled (GROK_API_KEY not set)")

# fal.ai LTX-2 Pro cloud models (requires fal.ai API key)
if settings.fal_api_key:
    models["Cloud: Kling v3"] = FalKlingTextToVideo()
    models["Cloud: Kling v3 I2V"] = FalKlingImageToVideo()
    models["Cloud: LTX-2 Pro"] = FalLTX2TextToVideo()
    models["Cloud: LTX-2 Pro I2V"] = FalLTX2ImageToVideo()
    logger.info("fal.ai cloud models enabled (Kling v3, LTX-2 Pro)")
else:
    logger.info("fal.ai cloud models disabled (FAL_API_KEY not set)")

# Local models always available
models["LTX-2"] = LTX2Model()
models["SVD-XT"] = SVDModel()
models["Wan 2.1"] = WanModel()

# Track which local model is currently loaded
_current_local_model: str | None = None


_SETTINGS_ENV_PATH = Path("/app/outputs/.settings.env")


def _write_env_file(updates: dict[str, str]) -> None:
    """Write settings to persistent .env on outputs volume."""
    env_path = _SETTINGS_ENV_PATH
    lines: list[str] = []
    existing_keys: set[str] = set()

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    existing_keys.add(key)
                    lines.append(f"{key}={updates[key]}")
                    continue
            lines.append(line)

    for key, value in updates.items():
        if key not in existing_keys:
            lines.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n")


def _load_persistent_settings() -> None:
    """Load saved settings from persistent volume on startup."""
    if not _SETTINGS_ENV_PATH.exists():
        return

    for line in _SETTINGS_ENV_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            continue
        # Only load keys not already set via environment
        if not os.environ.get(key):
            os.environ[key] = value
            if key == "GROK_API_KEY":
                settings.grok_api_key = value
            elif key == "FAL_API_KEY":
                settings.fal_api_key = value
            elif key == "NGC_API_KEY":
                settings.ngc_api_key = value

    # Register cloud models based on loaded keys
    if settings.grok_api_key and "Cloud: Grok Video" not in models:
        models["Cloud: Grok Video"] = GrokVideoModel()
        logger.info("Grok Video model enabled (from saved settings)")
    if settings.fal_api_key and "Cloud: LTX-2 Pro" not in models:
        models["Cloud: Kling v3"] = FalKlingTextToVideo()
        models["Cloud: Kling v3 I2V"] = FalKlingImageToVideo()
        models["Cloud: LTX-2 Pro"] = FalLTX2TextToVideo()
        models["Cloud: LTX-2 Pro I2V"] = FalLTX2ImageToVideo()
        logger.info("fal.ai cloud models enabled (from saved settings)")


# Load API keys saved from Settings UI (persisted on outputs volume)
_load_persistent_settings()


def get_available_model_choices() -> list[str]:
    """Return MODEL_CHOICES filtered to only models currently registered."""
    return [c for c in MODEL_CHOICES if c in models]


def _mask_key(key: str) -> str:
    """Mask an API key for display, showing only last 4 chars."""
    if not key:
        return ""
    if len(key) <= 4:
        return "****"
    return "****" + key[-4:]


def save_settings(
    grok_key: str,
    fal_key: str,
    ngc_key: str,
    models_dir_str: str,
    outputs_dir_str: str,
) -> tuple[str, dict]:
    """Save settings to .env, update runtime config, re-register cloud models.

    Returns (status_message, dropdown_update).
    """
    changes: list[str] = []
    env_updates: dict[str, str] = {}

    # Only update API keys if they were actually changed (not masked)
    if grok_key and not grok_key.startswith("****"):
        settings.grok_api_key = grok_key
        env_updates["GROK_API_KEY"] = grok_key
        os.environ["GROK_API_KEY"] = grok_key
    if fal_key and not fal_key.startswith("****"):
        settings.fal_api_key = fal_key
        env_updates["FAL_API_KEY"] = fal_key
        os.environ["FAL_API_KEY"] = fal_key
    if ngc_key and not ngc_key.startswith("****"):
        settings.ngc_api_key = ngc_key
        env_updates["NGC_API_KEY"] = ngc_key
        os.environ["NGC_API_KEY"] = ngc_key

    # Handle clearing keys (empty field when previously set)
    if grok_key == "" and settings.grok_api_key:
        settings.grok_api_key = ""
        env_updates["GROK_API_KEY"] = ""
        os.environ.pop("GROK_API_KEY", None)
    if fal_key == "" and settings.fal_api_key:
        settings.fal_api_key = ""
        env_updates["FAL_API_KEY"] = ""
        os.environ.pop("FAL_API_KEY", None)
    if ngc_key == "" and settings.ngc_api_key:
        settings.ngc_api_key = ""
        env_updates["NGC_API_KEY"] = ""
        os.environ.pop("NGC_API_KEY", None)

    # Update paths
    if models_dir_str:
        new_models_dir = Path(models_dir_str)
        if new_models_dir != settings.models_dir:
            settings.models_dir = new_models_dir
            env_updates["MODELS_DIR"] = models_dir_str
            changes.append(f"Models dir: {models_dir_str}")
    if outputs_dir_str:
        new_outputs_dir = Path(outputs_dir_str)
        if new_outputs_dir != settings.outputs_dir:
            settings.outputs_dir = new_outputs_dir
            env_updates["OUTPUTS_DIR"] = outputs_dir_str
            settings.outputs_dir.mkdir(parents=True, exist_ok=True)
            changes.append(f"Outputs dir: {outputs_dir_str}")

    # Re-register/unregister cloud models based on current keys
    if settings.grok_api_key and "Cloud: Grok Video" not in models:
        models["Cloud: Grok Video"] = GrokVideoModel()
        changes.append("Grok Video model enabled")
    elif not settings.grok_api_key and "Cloud: Grok Video" in models:
        del models["Cloud: Grok Video"]
        changes.append("Grok Video model disabled")

    if settings.fal_api_key:
        if "Cloud: LTX-2 Pro" not in models:
            models["Cloud: Kling v3"] = FalKlingTextToVideo()
            models["Cloud: Kling v3 I2V"] = FalKlingImageToVideo()
            models["Cloud: LTX-2 Pro"] = FalLTX2TextToVideo()
            models["Cloud: LTX-2 Pro I2V"] = FalLTX2ImageToVideo()
            changes.append("fal.ai cloud models enabled")
    else:
        removed_fal = False
        for fal_name in [
            "Cloud: Kling v3",
            "Cloud: Kling v3 I2V",
            "Cloud: LTX-2 Pro",
            "Cloud: LTX-2 Pro I2V",
        ]:
            if fal_name in models:
                del models[fal_name]
                removed_fal = True
        if removed_fal:
            changes.append("fal.ai cloud models disabled")

    # Persist to .env
    if env_updates:
        _write_env_file(env_updates)

    new_choices = get_available_model_choices()
    status = "Settings saved. " + (", ".join(changes) if changes else "No changes.")
    logger.info(status)

    return status, gr.Dropdown(choices=new_choices, value=new_choices[0] if new_choices else None)


def get_current_settings() -> tuple:
    """Return current settings values for UI refresh on page load."""
    choices = get_available_model_choices()
    return (
        gr.Dropdown(choices=choices, value="LTX-2" if "LTX-2" in choices else choices[0]),
        _mask_key(settings.grok_api_key),
        _mask_key(settings.fal_api_key),
        _mask_key(settings.ngc_api_key),
        str(settings.models_dir),
        str(settings.outputs_dir),
    )


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
        save_settings_fn=save_settings,
        get_model_choices_fn=get_available_model_choices,
        mask_key_fn=_mask_key,
        get_current_settings_fn=get_current_settings,
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
