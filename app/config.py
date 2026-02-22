"""Application configuration using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration with defaults optimized for NVIDIA DGX Spark (128GB unified)."""

    # Cosmos NIM
    cosmos_nim_url: str = "http://cosmos-nim:8000"
    cosmos_timeout_seconds: float = 600.0

    # Model paths
    models_dir: Path = Path("/app/models")
    outputs_dir: Path = Path("/app/outputs")

    # LTX-2 defaults
    ltx2_model_id: str = "Lightricks/LTX-2"
    ltx2_use_fp8: bool = False  # Not needed with bfloat16 on 128GB

    # SVD-XT defaults
    svd_model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"

    # Wan 2.1 defaults
    wan_model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

    # Generation defaults
    default_width: int = 768
    default_height: int = 512
    default_num_frames: int = 49
    default_fps: int = 24
    default_guidance_scale: float = 3.0
    default_num_inference_steps: int = 30

    # Resource management
    max_concurrent_requests: int = 1
    max_output_files: int = 100
    max_output_size_gb: float = 10.0

    # NGC
    ngc_api_key: str = ""

    # Grok Imagine (xAI)
    grok_api_key: str = ""
    grok_api_timeout: float = 600.0

    # fal.ai (LTX-2 Pro cloud)
    fal_api_key: str = ""
    fal_api_timeout: float = 600.0

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "extra": "ignore",
        "validate_assignment": True,
    }

    @property
    def ltx2_local_path(self) -> Path:
        return self.models_dir / "ltx2"

    @property
    def svd_local_path(self) -> Path:
        return self.models_dir / "svd"


settings = Settings()
