# AGENT.md

**Project Name:** Aethr  
**Full Title:** Aethr – Adaptive Aerial Defect Inspector for Solar Infrastructure  
**Version:** 1.0  
**Last Updated:** February 2026

## About the Project

**Aethr** is an end-to-end computer vision system designed to automatically detect and classify defects in solar photovoltaic (PV) panels using drone or aerial imagery.

The system focuses on common real-world defects found in solar installations and includes a built-in mechanism for continuous model improvement through human feedback.

### Key Features

- Object detection of the most frequent solar panel defects:
  - Cracks / fractures
  - Dust / soiling accumulation
  - Bird droppings
  - Physical damage (chips, breaks, delamination)
  - Hotspots (thermal anomalies – supported when thermal images are used)
- **Active learning / human-in-the-loop** feedback loop:
  - The model flags uncertain predictions
  - User corrects or confirms detections via a simple web interface
  - Corrected examples are collected and used to fine-tune the model periodically
- Modern MLOps practices:
  - Data versioning with DVC
  - Experiment tracking with Weights & Biases
  - Model export to ONNX format for faster / more portable inference
  - REST API with FastAPI
  - Containerization with Docker
- Interactive web dashboard (Streamlit):
  - Upload images → run inference
  - Visualize detections with confidence scores
  - Flag uncertain cases for review
  - Basic geospatial preview (simulated coordinates using Folium)
- Designed to run training on free GPU resources (Google Colab) while keeping development, inference and UI work on a local laptop

The architecture follows contemporary best practices for building maintainable, iterable computer vision pipelines suitable for inspection / monitoring applications.

## Tech Stack

- Model: Ultralytics YOLO11 (nano or small variants recommended)
- Framework: PyTorch + ONNX Runtime
- MLOps: DVC, Weights & Biases
- Backend API: FastAPI + Uvicorn
- Dashboard: Streamlit + Folium
- Containerization: Docker
- Training environment: Google Colab (free T4 GPU tier)
- Local development: VS Code

## Folder Structure

```
aethr/
├── AGENT.md                  ← This file
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml        (optional)
├── dvc.yaml
├── params.yaml
├── data/
│   ├── raw/                  ← Roboflow dataset folder is already here
│   ├── processed/            (generated if you run preprocessing)
│   └── feedback/             ← folder for collecting new annotations
├── notebooks/
│   ├── 01_train_yolo11.ipynb
│   ├── 02_evaluate.ipynb
│   └── 03_active_learning.ipynb
├── src/
│   ├── config.py
│   ├── data_preprocess.py
│   └── inference/
│       └── predictor.py
├── app/                      ← Streamlit application
│   └── streamlit_app.py
├── api/                      ← FastAPI application
│   └── main.py
├── models/                   ← trained weights (.pt, .onnx)
├── runs/                     ← training logs / artifacts (gitignored)
├── docs/
│   └── architecture.drawio.png  (create your own diagram)
└── .gitignore
```

## 1. Installation (Local – VS Code)

```bash
# 1. Go to project folder
cd aethr

# 2. Create and activate environment
conda create -n aethr python=3.10 -y
conda activate aethr

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize DVC (data versioning)
dvc init
# Optional: add a local remote or Google Drive remote
# dvc remote add -d gdrive gdrive://your-folder-id
```

## 2. Data Setup

You have already downloaded the Roboflow dataset and placed it locally in the data/ folder.

Expected structure inside `data/raw/`:

```
data/
└── raw/
    └── solar-panel-defects/           ← folder name can vary – use whatever Roboflow gave you
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        ├── test/                         (may or may not exist)
        └── data.yaml
```

Command to version the data with DVC:

```bash
# Replace 'solar-panel-defects' with your actual folder name if different
dvc add data/raw/solar-panel-defects
git add data/raw/solar-panel-defects.dvc .gitignore
git commit -m "Track Roboflow solar panel dataset"
```

Typical Roboflow splits (use these):

- `train/`   → 70–80% of images (model learning)
- `valid/`   → 15–20% (hyperparameter tuning / early stopping)
- `test/`    → remaining 5–10% (final unbiased evaluation – only used at the very end)

Do not manually re-split unless the dataset has serious issues.

## 3. Training – Using Google Colab (Free GPU)

Recommended workflow:

1. Open VS Code
2. Open `notebooks/01_train_yolo11.ipynb`
3. Use the Colab extension or click any "Open in Colab" badge/link
4. In Colab:
   - Change runtime → T4 GPU
   - Run cells step-by-step
   - The notebook will read the data.yaml from your repo (or you can upload it)

Typical training command (inside notebook):

```bash
!yolo task=detect mode=train model=yolo11s.pt data=/content/data/raw/solar-panel-defects/data.yaml epochs=80 imgsz=640 batch=16 project=/content/runs name=aethr
```

After training:

- Download the best model (`best.pt`) from Colab
- Move it to `models/` folder locally
- Optionally version it with DVC: `dvc add models/best.pt`

## 4. Core Local Commands

```bash
# Launch dashboard
streamlit run app/streamlit_app.py

# Launch inference API
cd api
uvicorn main:app --reload --port 8000
```

## 5. Docker (Local Testing)

```bash
# Build API container
docker build -t aethr-api .

# Run it
docker run -p 8000:8000 aethr-api
```

## 6. Active Learning / Feedback Loop

High-level flow:

1. Run Streamlit dashboard
2. Upload aerial images → see predictions + confidence scores
3. Flag low-confidence / incorrect detections
4. Corrections are saved into `data/feedback/`
5. When enough new examples are collected:
   - Run the active-learning notebook in Colab
   - Fine-tune starting from previous best model
   - Deploy updated weights

This creates a realistic iterative improvement cycle.
