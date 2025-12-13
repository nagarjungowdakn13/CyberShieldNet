# Reproducibility

This document provides environment details and commands to replicate experiments and generate evaluation artifacts.

## Environment

- OS: Ubuntu 20.04 or Windows 10/11
- Python: 3.10.x
- GPU: NVIDIA V100 (optional). CPU-only supported with simplified flow.
- Key packages (see `requirements.txt`): torch, torch-geometric, numpy, pandas, scikit-learn, fastapi, uvicorn, shap, captum.

## Setup

```
pip install -e .
```

Optionally create a virtual environment.

## Training

Minimal smoke training (will fallback to sklearn without torch):

```
python src\scripts\train_model.py --epochs 2 --model-out data\models\cybershieldnet.pth
```

## Evaluation

Use the evaluation harness to compute metrics and save plots.

```
python experiments\run_evaluation.py --config config\model_config.yaml --results-out data\results\metrics.json
```

Artifacts:

- `data/results/metrics.json`: accuracy/precision/recall/AUC
- `data/results/roc.png`, `data/results/pr.png`: ROC and Precision-Recall plots

## Notes

- Ensure dataset configuration paths in `config/data_config.yaml` are set.
- For CPU-only environments, reduce batch sizes and graph sizes where necessary.
