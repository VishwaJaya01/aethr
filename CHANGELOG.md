# Changelog

All notable changes to Aethr are documented in this file.

## [1.0.0] - 2026-02-14

### Added

- End-to-end YOLO11 baseline project scaffold.
- FastAPI inference service in `api/main.py`.
- Streamlit dashboard in `app/streamlit_app.py`.
- Core inference and config modules under `src/`.
- Colab-ready notebooks for training, evaluation, and active learning.
- Docker and Docker Compose local runtime configuration.
- Project metrics baseline document in `docs/metrics_v1.md`.
- Explicit project version file (`VERSION`).

### Model and Metrics

- Baseline model artifacts produced: `models/best.pt`, `models/best.onnx`.
- Reported baseline quality: `mAP50-95 = 0.314` (YOLO11s, v1 dataset run).

### Data and Feedback Workflow

- Data policy enforced through `.gitignore` to prevent raw dataset commits.
- Feedback collection implemented in Streamlit and saved to `data/feedback/`.
- Active-learning path documented for future re-annotation and fine-tuning.
