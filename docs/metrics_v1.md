# Aethr Metrics v1

- Date: 2026-02-14
- Model: YOLO11s (`yolo11s.pt` baseline)
- Training Environment: Google Colab Web (T4 GPU)
- Dataset Version: `solar-panel-defects-v1`
- Task: Object Detection (7 classes)
- Split: train/val/test = 75/20/5

## Summary

- Primary baseline metric: **mAP50-95 = 0.314**
- Source: user-reported result after baseline training (~70 epochs)
- Status: baseline complete, ready for class-wise error analysis and active learning loop

## Validation Metrics

- mAP50-95: `0.314`
- mAP50: `TBD`
- Precision: `TBD`
- Recall: `TBD`

## Test Metrics

- mAP50-95: `TBD`
- mAP50: `TBD`
- Precision: `TBD`
- Recall: `TBD`

## Per-Class Metrics (Validation)

| Class Name | Precision | Recall | mAP50 | mAP50-95 | Notes |
|---|---:|---:|---:|---:|---|
| class_1 | TBD | TBD | TBD | TBD | |
| class_2 | TBD | TBD | TBD | TBD | |
| class_3 | TBD | TBD | TBD | TBD | |
| class_4 | TBD | TBD | TBD | TBD | |
| class_5 | TBD | TBD | TBD | TBD | |
| class_6 | TBD | TBD | TBD | TBD | |
| class_7 | TBD | TBD | TBD | TBD | |

## Training Configuration Snapshot

- Epochs run: `~70` (user-reported)
- Image size: `640`
- Batch size: `16`
- Early stop/patience: `TBD`
- Confidence threshold at inference: `0.25` (project default)

## Artifact Paths

- Best PyTorch model: `models/best.pt`
- Best ONNX model: `models/best.onnx`
- Colab/Drive runs directory: `.../aethr/runs/`

## How To Fill Remaining Metrics From Notebook 02

Run this in `notebooks/02_evaluate.ipynb` after `model.val(...)`:

```python
print('VAL metrics:', val_metrics.results_dict)

# If test split exists
# print('TEST metrics:', test_metrics.results_dict)
```

Then copy values into this file.

## Next Actions

1. Fill missing validation/test metrics from notebook 02 outputs.
2. Replace `class_1..class_7` with exact class names from `data.yaml`.
3. Run targeted error analysis for lowest-performing classes.
4. Start active learning cycle after collecting feedback samples in `data/feedback/`.
