# Aethr

Aethr is an adaptive aerial defect inspector for solar infrastructure.
It detects common PV defects from drone imagery using YOLO11 and supports
active learning with human feedback.

## Scope

- Defects: cracks, dust/soiling, bird droppings, physical damage, hotspots.
- Model family: Ultralytics YOLO11 (`yolo11n.pt` / `yolo11s.pt`).
- Training: Google Colab free T4 GPU (preferred).
- Local: FastAPI API + Streamlit dashboard + DVC/W&B tracking.

## Project Structure

```text
aethr/
├── AGENT.md
├── README.md
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
│   └── inference/
│       └── predictor.py
├── app/
│   └── streamlit_app.py
├── api/
│   └── main.py
├── models/
├── runs/
└── docs/
    └── architecture.drawio.png
```

## Setup (Local)

```bash
conda create -n aethr python=3.10 -y
conda activate aethr
pip install -r requirements.txt
```

## Training (Google Colab Extension Workflow)

1. Open this repo in VS Code.
2. Open `notebooks/01_train_yolo11.ipynb`.
3. Use the VS Code Google Colab extension to run notebook cells in Colab.
4. Set runtime to **GPU (T4)**.
5. Run cells top-to-bottom.

Notes:
- Keep dataset under `data/raw/.../data.yaml`.
- Save trained `best.pt` to `models/best.pt`.
- Export ONNX to `models/best.onnx`.

## Run Locally

API:

```bash
uvicorn api.main:app --reload --port 8000
```

Dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Active Learning Loop

1. Upload image in Streamlit.
2. Review predictions and uncertain detections.
3. Save feedback package to `data/feedback/`.
4. Re-annotate feedback set and fine-tune with
   `notebooks/03_active_learning.ipynb` on Colab.
