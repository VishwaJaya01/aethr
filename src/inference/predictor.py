"""Inference utilities for YOLO11-based solar defect detection."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ultralytics import YOLO

LOGGER = logging.getLogger("aethr.predictor")


@dataclass(slots=True)
class Detection:
    """Structured representation of a single detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    uncertain: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert detection to serializable dict."""
        return asdict(self)


class Predictor:
    """Wrapper around Ultralytics YOLO inference for app/API usage."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        uncertainty_threshold: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
        device: Optional[str | int] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.conf_threshold = conf_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = device

        LOGGER.info("Loading model: %s", self.model_path)
        self.model = YOLO(str(self.model_path))

    def _resolve_class_name(self, class_id: int, names: Any) -> str:
        """Resolve class name from result names mapping/list."""
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def predict(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> list[Detection]:
        """Run inference and return typed detections."""
        conf = self.conf_threshold if conf_threshold is None else conf_threshold

        try:
            results = self.model.predict(
                source=image,
                conf=conf,
                imgsz=self.imgsz,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            raise RuntimeError(f"Prediction failed: {exc}") from exc

        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        detections: list[Detection] = []
        names = getattr(result, "names", None) or getattr(self.model, "names", None)

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            coords = [float(value) for value in box.xyxy.squeeze().tolist()]
            class_name = self._resolve_class_name(class_id, names)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=coords,
                    uncertain=confidence < self.uncertainty_threshold,
                )
            )

        return detections

    def plot(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Return rendered image with boxes in BGR format."""
        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        results = self.model.predict(
            source=image,
            conf=conf,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        if not results:
            raise RuntimeError("No result returned from model.")

        return results[0].plot()

    @staticmethod
    def uncertain_count(detections: list[Detection]) -> int:
        """Count low-confidence detections flagged for review."""
        return sum(1 for detection in detections if detection.uncertain)
