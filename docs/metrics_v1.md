# Aethr Metrics v1

- Release: `v1.0.0`
- Date: 2026-02-14
- Model: YOLO11s (`yolo11s.pt` baseline)
- Training Environment: Google Colab Web (T4 GPU)
- Dataset Version: `solar-panel-defects-v1`
- Task: Object Detection (7 classes)
- Split: train/val/test = 75/20/5

## Summary

- Primary baseline metric: **mAP50-95 = 0.314**
- Source: user-reported result after baseline training (~70 epochs)
- Status: baseline v1 complete, ready for class-wise error analysis and active learning loop

## Validation Metrics

- mAP50-95: `0.314`
- mAP50: `Not recorded in v1 notes`
- Precision: `Not recorded in v1 notes`
- Recall: `Not recorded in v1 notes`

## Test Metrics

- mAP50-95: `Not recorded in v1 notes`
- mAP50: `Not recorded in v1 notes`
- Precision: `Not recorded in v1 notes`
- Recall: `Not recorded in v1 notes`

## Per-Class Metrics (Validation)

| Class Name | Precision | Recall | mAP50 | mAP50-95 | Notes |
|---|---:|---:|---:|---:|---|
| class_1 | Not recorded | Not recorded | Not recorded | Not recorded | Replace with real class name |
| class_2 | Not recorded | Not recorded | Not recorded | Not recorded | Replace with real class name |
| class_3 | Not recorded | Not recorded | Not recorded | Not recorded | Replace with real class name |
| class_4 | Not recorded | Not recorded | Not recorded | Not recorded | Replace with real class name |
| class_5 | Not recorded | Not recorded | Not recorded | Not recorded | Replace with real class name |
| class_6 | Not recorded | Not recorded | Not recorded | Not recorded | Replace with real class name |
| class_7 | Not recorded | Not recorded | Not recorded | Not recorded | Replace with real class name |

## Training Configuration Snapshot

- Epochs run: `~70` (user-reported)
- Image size: `640`
- Batch size: `16`
- Early stop/patience: `Not recorded in v1 notes`
- Confidence threshold at inference: `0.25` (project default)

## Artifact Paths

- Best PyTorch model: `models/best.pt`
- Best ONNX model: `models/best.onnx`
- Colab/Drive runs directory: `.../aethr/runs/`

## Optional Backfill (If You Want Full Metric Table)

Run this in `notebooks/02_evaluate.ipynb` after `model.val(...)` to backfill missing fields:

```python
print('VAL metrics:', val_metrics.results_dict)

# If test split exists
# print('TEST metrics:', test_metrics.results_dict)
```

Then copy values into this file.

## Next Actions

1. Replace `class_1..class_7` with exact class names from `data.yaml`.
2. Optionally fill remaining validation/test metrics from notebook 02 outputs.
3. Run targeted error analysis for lowest-performing classes.
4. Start active learning cycle after collecting feedback samples in `data/feedback/`.
