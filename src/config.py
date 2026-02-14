"""Centralized configuration for Aethr."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Runtime settings derived from env vars and project defaults."""

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    feedback_dir: Path = PROJECT_ROOT / "data" / "feedback"
    models_dir: Path = PROJECT_ROOT / "models"
    runs_dir: Path = PROJECT_ROOT / "runs"

    default_model_path: Path = PROJECT_ROOT / "models" / "best.pt"
    default_model_variant: str = os.getenv("AETHR_MODEL_VARIANT", "yolo11s.pt")

    conf_threshold: float = float(os.getenv("AETHR_CONF_THRESHOLD", "0.25"))
    uncertainty_threshold: float = float(
        os.getenv("AETHR_UNCERTAINTY_THRESHOLD", "0.45")
    )
    imgsz: int = int(os.getenv("AETHR_IMGSZ", "640"))
    max_det: int = int(os.getenv("AETHR_MAX_DET", "300"))

    api_host: str = os.getenv("AETHR_API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("AETHR_API_PORT", "8000"))


def get_settings() -> Settings:
    """Return initialized settings and ensure key folders exist."""
    settings = Settings()
    settings.feedback_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    return settings
