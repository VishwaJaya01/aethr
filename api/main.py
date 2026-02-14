"""FastAPI service for Aethr model inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.inference.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("aethr.api")

settings = get_settings()
app = FastAPI(
    title="Aethr API",
    version="1.0.0",
    description="Solar panel defect detection API powered by YOLO11.",
)

_predictor: Optional[Predictor] = None


class PredictionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    uncertain: bool


class PredictResponse(BaseModel):
    detections: list[PredictionItem]
    uncertain_count: int
    model_path: str


def resolve_model_path() -> Path:
    """Resolve model path from env var or default settings."""
    configured_path = os.getenv("AETHR_MODEL_PATH", "").strip()
    if configured_path:
        return Path(configured_path)
    return Path(settings.default_model_path)


def get_predictor() -> Predictor:
    """Lazy-load model to keep startup fast and fail gracefully."""
    global _predictor
    if _predictor is not None:
        return _predictor

    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train in Colab and copy best.pt to models/."
        )

    _predictor = Predictor(
        model_path=model_path,
        conf_threshold=settings.conf_threshold,
        uncertainty_threshold=settings.uncertainty_threshold,
        imgsz=settings.imgsz,
        max_det=settings.max_det,
    )
    return _predictor


@app.get("/")
def root() -> dict[str, str]:
    """Simple API metadata endpoint."""
    return {"service": "aethr-api", "status": "ready"}


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Health endpoint with model availability status."""
    model_path = resolve_model_path()
    return {
        "status": "ok",
        "model_exists": model_path.exists(),
        "model_path": str(model_path),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> PredictResponse:
    """Run inference on an uploaded image and return detections."""
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    temp_path: Optional[Path] = None

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            payload = await file.read()
            temp_file.write(payload)
            temp_path = Path(temp_file.name)

        detections = predictor.predict(temp_path, conf_threshold=conf)
        response_items = [item.to_dict() for item in detections]

        return PredictResponse(
            detections=response_items,
            uncertain_count=predictor.uncertain_count(detections),
            model_path=str(predictor.model_path),
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
