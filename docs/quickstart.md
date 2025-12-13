# Quick Start

This guide helps you install, train a small model, and run the API.

## Prerequisites

- Python 3.10+
- Optional: NVIDIA GPU + CUDA for faster training
- Windows users: run commands in `cmd.exe`

## Install

```
pip install -e .
```

## Train (smoke run)

Runs a tiny training job. Falls back to sklearn if PyTorch isn't available.

```
python src\scripts\train_model.py --epochs 1 --model-out data\models\cybershieldnet.pth
```

## Run API (local)

```
uvicorn src.cybershieldnet.api.fastapi:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and `http://localhost:8000/docs`.

## Docker (CPU or GPU)

Build image and start services:

```
docker build -t cybershieldnet .
docker compose up -d
```

Note: If no NVIDIA GPU is present, remove the `deploy.resources.reservations.devices` sections in `docker-compose.yml`.

## Next Steps

- See `docs/deployment_guide.md` for production notes.
- See `docs/reproducibility.md` for experiment environments and evaluation.
