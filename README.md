# Aethr

Adaptive Aerial Defect Inspector for solar infrastructure using drone and aerial imagery.

**Version:** `1.0.0`  
**Release Status:** Baseline complete and production-oriented for portfolio demonstration.

## Overview

Aethr is an end-to-end computer vision project that detects common solar panel defects and provides a practical review loop for model improvement.  
The project combines model training in Google Colab with local inference services for API and dashboard usage.

## Core Capabilities

- Defect detection with Ultralytics YOLO11.
- Colab-first training and evaluation workflow with T4 GPU.
- ONNX export for portable, efficient inference.
- FastAPI service for model inference endpoints.
- Streamlit dashboard for visual inspection and reviewer feedback capture.
- Human-in-the-loop feedback packaging for active learning iterations.

## Baseline Performance (v1)

- Model: `YOLO11s`
- Dataset: 7-class solar defect detection dataset
- Reported metric: `mAP50-95 = 0.314`
- Detailed summary: `docs/metrics_v1.md`

## Technology Stack

- Model: Ultralytics YOLO11, PyTorch, ONNX Runtime
- Backend: FastAPI, Uvicorn
- UI: Streamlit, Folium
- MLOps and tooling: Docker, DVC (configured), Google Colab

## Repository Structure

```text
aethr/
├── AGENT.md
├── README.md
├── CHANGELOG.md
├── VERSION
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── feedback/
├── notebooks/
│   ├── 01_train_yolo11.ipynb
│   ├── 02_evaluate.ipynb
│   └── 03_active_learning.ipynb
├── src/
│   ├── config.py
│   ├── data_preprocess.py
│   └── inference/predictor.py
├── api/main.py
├── app/streamlit_app.py
├── models/
└── docs/
    ├── architecture.drawio.png
    └── metrics_v1.md
```

## Local Setup

### Windows (PowerShell)

```powershell
cd D:\Projects\aethr
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
cd aethr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data Requirements

Expected YOLO dataset layout under `data/raw/<dataset-name>/`:

```text
data.yaml
train/images
train/labels
valid/images
valid/labels
test/images    (optional)
test/labels    (optional)
```

## Training Workflow (Google Colab Web + Drive)

1. Open `notebooks/01_train_yolo11.ipynb` in Google Colab Web.
2. Set runtime to `GPU (T4)`.
3. Mount Google Drive in the notebook.
4. Point notebook variables to your Drive dataset path.
5. Run all cells to train and export artifacts.
6. Save `best.pt` and `best.onnx` in `models/`.

## Evaluation Workflow

Run `notebooks/02_evaluate.ipynb` after training to validate model quality.

- Evaluates on validation split.
- Evaluates on test split when defined in `data.yaml`.

## Local Inference

### Run API

```bash
uvicorn api.main:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Feedback and Active Learning

In Streamlit:

1. Upload image and run inference.
2. Mark uncertain or incorrect cases for review.
3. Save feedback package to `data/feedback/`.

Feedback output includes:

- Image copy
- JSON metadata (detections, uncertainty count, reviewer notes)

To fine-tune from feedback, re-annotate selected feedback images and export a YOLO dataset before running `notebooks/03_active_learning.ipynb`.

## Data and Artifact Policy

- Raw data under `data/` is intentionally ignored by Git.
- Keep only placeholders (`.gitkeep`) and optional DVC pointer files.
- Keep `runs/` local and untracked.

## Release and Versioning

- Current release: `VERSION`
- Release history: `CHANGELOG.md`
