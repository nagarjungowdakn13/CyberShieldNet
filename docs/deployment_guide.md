# Deployment Guide

This guide summarizes production-oriented deployment for CyberShieldNet.

## Environments

- Python 3.10, FastAPI, Uvicorn
- Optional GPU: CUDA 11.8, NVIDIA drivers
- Observability: Prometheus + Grafana (optional)

## Configuration

Project configs are under `config/`:

- `model_config.yaml`: model hyperparameters/fusion settings
- `training_config.yaml`: epochs, batch size, optimizer
- `data_config.yaml`: paths for raw/processed data
- `deployment_config.yaml`: API settings, CORS, trusted hosts

## Container Deployment

1. Build: `docker build -t cybershieldnet .`
2. Run stack: `docker compose up -d`
3. Health: `http://localhost:8000/health` (or `/` and `/docs`)

GPU notes:

- If host lacks NVIDIA GPU, remove `deploy.resources.reservations.devices` blocks in `docker-compose.yml` for API and worker.

## Security & Ops

- Network: restrict exposed ports; prefer reverse proxy and TLS
- Secrets: use environment variables or vault; avoid committing secrets
- Logs: container volume `./logs` is mounted; integrate with SIEM
- Scalability: run multiple API replicas behind a load balancer

## Monitoring

- Prometheus: configure scrape via `monitoring/prometheus.yml`
- Grafana: default admin password `admin` (change in env)

## Upgrades

- Pin dependency versions; test in staging
- Maintain model artifacts under `data/models` with versioned filenames

## Troubleshooting

- Uvicorn fails to start: verify `uvicorn src.cybershieldnet.api.fastapi:app`
- CUDA errors: ensure compatible driver/toolkit
- High memory: reduce batch sizes or graph size thresholds
