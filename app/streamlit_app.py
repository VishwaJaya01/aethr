"""Streamlit dashboard for Aethr inference and feedback collection."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import random
import sys

import folium
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.inference.predictor import Detection, Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("aethr.streamlit")


@st.cache_resource(show_spinner=False)
def load_predictor(
    model_path: str,
    conf_threshold: float,
    uncertainty_threshold: float,
    imgsz: int,
    max_det: int,
) -> Predictor:
    """Cache model instance between Streamlit reruns."""
    return Predictor(
        model_path=model_path,
        conf_threshold=conf_threshold,
        uncertainty_threshold=uncertainty_threshold,
        imgsz=imgsz,
        max_det=max_det,
    )


def detection_table(detections: list[Detection]) -> pd.DataFrame:
    """Build DataFrame for display and CSV export."""
    rows = []
    for idx, item in enumerate(detections, start=1):
        rows.append(
            {
                "id": idx,
                "class": item.class_name,
                "confidence": round(item.confidence, 4),
                "uncertain": item.uncertain,
                "x1": round(item.bbox_xyxy[0], 2),
                "y1": round(item.bbox_xyxy[1], 2),
                "x2": round(item.bbox_xyxy[2], 2),
                "y2": round(item.bbox_xyxy[3], 2),
            }
        )
    return pd.DataFrame(rows)


def render_simulated_map(detections: list[Detection], lat: float, lon: float) -> None:
    """Render a basic folium map with simulated defect pins near a base point."""
    fmap = folium.Map(location=[lat, lon], zoom_start=18, control_scale=True)
    rng = random.Random(7)

    for index, detection in enumerate(detections, start=1):
        lat_offset = (rng.random() - 0.5) * 0.0006
        lon_offset = (rng.random() - 0.5) * 0.0006
        folium.Marker(
            location=[lat + lat_offset, lon + lon_offset],
            tooltip=f"{index}. {detection.class_name}",
            popup=(
                f"class={detection.class_name}, "
                f"conf={detection.confidence:.2f}, "
                f"uncertain={detection.uncertain}"
            ),
        ).add_to(fmap)

    components.html(fmap._repr_html_(), height=420)


def save_feedback(
    feedback_dir: Path,
    image_bytes: bytes,
    original_name: str,
    detections: list[Detection],
    note: str,
) -> tuple[Path, Path]:
    """Persist image and metadata JSON for active-learning review."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = Path(original_name).stem or "uploaded"
    ext = Path(original_name).suffix or ".jpg"

    image_path = feedback_dir / f"{stem}_{timestamp}{ext}"
    json_path = feedback_dir / f"{stem}_{timestamp}.json"

    image_path.write_bytes(image_bytes)

    payload = {
        "created_at_utc": timestamp,
        "source_image": original_name,
        "saved_image": image_path.name,
        "note": note.strip(),
        "detections": [item.to_dict() for item in detections],
        "uncertain_count": Predictor.uncertain_count(detections),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return image_path, json_path


def initialize_state() -> None:
    """Initialize keys used across reruns."""
    st.session_state.setdefault("detections", [])
    st.session_state.setdefault("annotated_image", None)


def main() -> None:
    """Run Streamlit app."""
    st.set_page_config(page_title="Aethr Inspector", layout="wide")
    settings = get_settings()
    initialize_state()

    st.title("Aethr - Adaptive Aerial Defect Inspector")
    st.caption(
        "Colab-first training workflow: run notebooks with VS Code Google Colab extension "
        "on free T4 GPU."
    )

    with st.sidebar:
        st.header("Inference Settings")
        model_path = st.text_input(
            "Model path",
            value=str(settings.default_model_path),
            help="Train on Colab and copy best.pt to models/.",
        )
        conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
        uncertainty_threshold = st.slider("Uncertainty threshold", 0.05, 0.95, 0.45, 0.01)
        base_lat = st.number_input("Base latitude", value=6.9271, format="%.6f")
        base_lon = st.number_input("Base longitude", value=79.8612, format="%.6f")

    try:
        predictor = load_predictor(
            model_path=model_path,
            conf_threshold=conf_threshold,
            uncertainty_threshold=uncertainty_threshold,
            imgsz=settings.imgsz,
            max_det=settings.max_det,
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload aerial image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    )

    if uploaded_file is None:
        st.info("Upload an image to run detection.")
        return

    image_bytes = uploaded_file.getvalue()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(pil_image)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Input")
        st.image(pil_image, use_container_width=True)

    if st.button("Run Inference", type="primary"):
        with st.spinner("Running YOLO11 inference..."):
            detections = predictor.predict(np_image, conf_threshold=conf_threshold)
            plotted_bgr = predictor.plot(np_image, conf_threshold=conf_threshold)
            plotted_rgb = plotted_bgr[:, :, ::-1]
            st.session_state["detections"] = detections
            st.session_state["annotated_image"] = plotted_rgb

    detections: list[Detection] = st.session_state.get("detections", [])
    annotated_image = st.session_state.get("annotated_image")

    with col_right:
        st.subheader("Detections")
        if annotated_image is not None:
            st.image(annotated_image, use_container_width=True)
        else:
            st.info("Click 'Run Inference' to view predictions.")

    if detections:
        uncertain = Predictor.uncertain_count(detections)
        st.metric("Detections", len(detections))
        st.metric("Uncertain", uncertain)

        table = detection_table(detections)
        st.dataframe(table, use_container_width=True)

        st.subheader("Simulated Geospatial Preview")
        render_simulated_map(detections, lat=base_lat, lon=base_lon)

        st.subheader("Feedback")
        mark_for_feedback = st.checkbox(
            "Mark this sample for feedback review",
            value=uncertain > 0,
            help="This stores the image and detection metadata in data/feedback/.",
        )
        note = st.text_area(
            "Reviewer note",
            placeholder="Example: false positive on dust; crack box too wide.",
        )

        if st.button("Save Feedback Package"):
            if not mark_for_feedback:
                st.warning("Enable feedback checkbox to avoid accidental saves.")
            else:
                image_path, json_path = save_feedback(
                    feedback_dir=settings.feedback_dir,
                    image_bytes=image_bytes,
                    original_name=uploaded_file.name,
                    detections=detections,
                    note=note,
                )
                st.success(
                    "Saved feedback package: "
                    f"{image_path.name} and {json_path.name}"
                )
    else:
        st.warning("No detections yet. Run inference first.")


if __name__ == "__main__":
    main()
