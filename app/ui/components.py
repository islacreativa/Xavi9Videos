"""Gradio UI components for Xavi9Videos."""

from __future__ import annotations

import json

import gradio as gr

from app.config import settings

MODEL_CHOICES = [
    "Cosmos Text2World",
    "Cosmos Video2World",
    "Cloud: Grok Video",
    "Cloud: LTX-2 Pro",
    "Cloud: LTX-2 Pro I2V",
    "LTX-2",
    "SVD-XT",
    "Wan 2.1",
]

# Models that require an image
IMAGE_REQUIRED_MODELS = {"Cosmos Video2World", "Cloud: LTX-2 Pro I2V", "SVD-XT"}
# Models that support text prompts
TEXT_MODELS = {
    "Cosmos Text2World",
    "Cosmos Video2World",
    "Cloud: Grok Video",
    "Cloud: LTX-2 Pro",
    "LTX-2",
    "Wan 2.1",
}
# Models with fixed resolution
FIXED_RESOLUTION_MODELS = {"SVD-XT"}


def update_ui_visibility(model_name: str) -> dict:
    """Return visibility updates based on selected model."""
    needs_image = model_name in IMAGE_REQUIRED_MODELS
    has_text = model_name in TEXT_MODELS
    has_resolution = model_name not in FIXED_RESOLUTION_MODELS
    is_svd = model_name == "SVD-XT"

    return {
        "prompt_visible": has_text,
        "image_visible": needs_image or model_name in {"LTX-2", "Cloud: Grok Video"},
        "image_required": needs_image,
        "resolution_visible": has_resolution,
        "frames_visible": not is_svd,
        "fps_visible": not is_svd,
        "guidance_visible": not is_svd,
    }


def _format_json(data) -> str:
    """Format dict/list as indented JSON string for display."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, ensure_ascii=False)


def build_ui(generate_fn, health_check_fn):
    """Build and return the Gradio Blocks interface."""

    # Wrap generate_fn to convert duration to num_frames and add progress
    async def wrapped_generate(
        progress=gr.Progress(track_tqdm=True),
        model_name="",
        prompt="",
        image=None,
        width=768,
        height=512,
        duration_seconds=2.0,
        fps=24,
        num_inference_steps=30,
        guidance_scale=3.0,
        seed=-1,
    ):
        def progress_callback(step, total):
            progress(step / total, desc=f"Step {step}/{total}")

        # Convert duration (seconds) to num_frames
        num_frames = max(9, round(duration_seconds * fps) + 1)

        result = await generate_fn(
            model_name,
            prompt,
            image,
            width,
            height,
            num_frames,
            fps,
            num_inference_steps,
            guidance_scale,
            seed,
            progress_callback=progress_callback,
        )
        video_path, metadata, status = result
        return video_path, _format_json(metadata), status

    # Wrap health_check to return JSON string
    async def wrapped_health():
        result = await health_check_fn()
        return gr.update(value=_format_json(result), visible=True)

    with gr.Blocks(
        title="Xavi9Videos - AI Video Generation",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Xavi9Videos\nAI Video Generation on NVIDIA DGX Spark | Grok - LTX-2 - SVD-XT - Wan 2.1"
        )

        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="LTX-2",
                    label="Model",
                    interactive=True,
                )

                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3,
                )

                image_input = gr.Image(
                    label="Input Image (optional for LTX-2, required for SVD/Cosmos V2W)",
                    type="pil",
                )

                with gr.Accordion("Generation Parameters", open=False):
                    with gr.Row():
                        width_input = gr.Slider(
                            256,
                            1920,
                            value=settings.default_width,
                            step=64,
                            label="Width",
                        )
                        height_input = gr.Slider(
                            256,
                            1080,
                            value=settings.default_height,
                            step=64,
                            label="Height",
                        )

                    duration_input = gr.Slider(
                        0.5,
                        10.0,
                        value=round(settings.default_num_frames / settings.default_fps, 1),
                        step=0.5,
                        label="Duration (seconds)",
                    )

                    with gr.Row():
                        fps_input = gr.Slider(
                            7,
                            50,
                            value=settings.default_fps,
                            step=1,
                            label="FPS",
                        )
                        steps_input = gr.Slider(
                            10,
                            100,
                            value=settings.default_num_inference_steps,
                            step=5,
                            label="Inference Steps",
                        )

                    guidance_input = gr.Slider(
                        1.0,
                        15.0,
                        value=settings.default_guidance_scale,
                        step=0.5,
                        label="Guidance Scale",
                    )

                    seed_input = gr.Number(
                        value=-1,
                        label="Seed (-1 = random)",
                        precision=0,
                    )

                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

                with gr.Row():
                    health_btn = gr.Button("Check Model Health", size="sm")

                health_output = gr.Textbox(
                    label="Model Health",
                    visible=False,
                    interactive=False,
                    lines=8,
                )

            # Right column: Output
            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video")

                with gr.Accordion("Generation Info", open=False):
                    info_output = gr.Textbox(
                        label="Metadata",
                        interactive=False,
                        lines=6,
                    )

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=3,
                )

        # Event handlers
        def on_model_change(model_name):
            vis = update_ui_visibility(model_name)
            return [
                gr.update(visible=vis["prompt_visible"]),
                gr.update(
                    visible=vis["image_visible"],
                    label=(
                        "Input Image (required)"
                        if vis["image_required"]
                        else "Input Image (optional)"
                    ),
                ),
                gr.update(visible=vis["resolution_visible"]),
                gr.update(visible=vis["resolution_visible"]),
                gr.update(visible=vis["frames_visible"]),
                gr.update(visible=vis["fps_visible"]),
                gr.update(visible=vis["guidance_visible"]),
            ]

        model_selector.change(
            fn=on_model_change,
            inputs=[model_selector],
            outputs=[
                prompt_input,
                image_input,
                width_input,
                height_input,
                duration_input,
                fps_input,
                guidance_input,
            ],
        )

        generate_btn.click(
            fn=wrapped_generate,
            inputs=[
                model_selector,
                prompt_input,
                image_input,
                width_input,
                height_input,
                duration_input,
                fps_input,
                steps_input,
                guidance_input,
                seed_input,
            ],
            outputs=[video_output, info_output, status_output],
        )

        health_btn.click(
            fn=wrapped_health,
            inputs=[],
            outputs=[health_output],
        )

    return demo
