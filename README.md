# Aethr

Adaptive Aerial Defect Inspector for solar infrastructure using drone/aerial imagery.

**Release:** `v1.0.0`  
**Status:** Baseline complete and runnable end-to-end.

## What v1 Includes

- YOLO11 detection pipeline for solar panel defects.
- Google Colab Web + T4 training workflow.
- Evaluation notebook for validation/test checks.
- FastAPI inference service.
- Streamlit dashboard with uncertainty-aware feedback capture.
- ONNX export for portable inference.

## Baseline Result (v1)

- Model: `YOLO11s`
- Dataset: Roboflow object detection dataset (7 classes)
- Reported baseline metric: `mAP50-95 = 0.314`
- Metrics document: `docs/metrics_v1.md`

## Project Structure

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

## Setup (Local)

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

## Training (Google Colab Web + Google Drive)

1. Open `notebooks/01_train_yolo11.ipynb` in Google Colab Web.
2. Set runtime to `GPU (T4)`.
3. Mount Drive in notebook.
4. Point dataset path to Drive YOLO export with `data.yaml`.
5. Run training cells top-to-bottom.
6. Save artifacts to Drive and copy to local `models/` as needed.

Required dataset layout:

```text
.../data/raw/<dataset-name>/
  data.yaml
  train/images
  train/labels
  valid/images
  valid/labels
  test/images  (optional)
  test/labels  (optional)
```

## Evaluation

Run `notebooks/02_evaluate.ipynb` after training.

- Uses `models/best.pt` and dataset `data.yaml`.
- Reports validation metrics and test metrics (if test split exists).

## Local Inference

### API

```bash
uvicorn api.main:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Feedback Collection (Human-in-the-Loop)

In Streamlit:

1. Upload image.
2. Run inference.
3. Enable `Mark this sample for feedback review`.
4. Add review notes.
5. Click `Save Feedback Package`.

Saved output:

- Feedback image copy under `data/feedback/`
- JSON metadata package under `data/feedback/`

Note: feedback JSON is metadata only. Fine-tuning requires re-annotation and a new YOLO dataset export.

## Data and Artifact Policy

- Do not commit raw dataset files in `data/`.
- Keep only placeholders (`.gitkeep`) and optional DVC pointers.
- `runs/` is local experiment output and should remain untracked.

## Versioning

- Current release version is stored in `VERSION`.
- Release changes are tracked in `CHANGELOG.md`.
