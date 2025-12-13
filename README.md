# CyberShieldNet: Multi-Modal Fusion Framework for Predictive Threat Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

## Overview

CyberShieldNet is a multi-modal fusion framework for predictive cyber threat intelligence. It integrates heterogeneous data sources and adaptive machine learning ensembles to assess risk and support security operations.

### Key Capabilities

- High prediction accuracy on security events
- Reduction in false positives compared to baseline methods
- Real-time processing capability at production event rates
- Explainability support with SHAP and attention mechanisms
- Adversarial robustness against evasion and poisoning attacks

## Quick Start

### Installation

```bash
git clone https://github.com/nagarjungowda/CyberShieldNet.git
cd CyberShieldNet
pip install -e .
```

### Usage (Illustrative)

```python
from cybershieldnet import CyberShieldNet

model = CyberShieldNet(config_path='config/model_config.yaml')
model.fit(training_data)
predictions = model.predict(security_events)
risk_scores = model.assess_risk(predictions)
```

### Documentation

- Installation Guide
- Quick Start
- API Reference
- Deployment Guide

### Dataset

CyberShieldNet-CTI benchmark dataset: CyberShieldNet-Data

### Citation

If you use CyberShieldNet in your research, please cite:

```
@article{gowda2024cybershieldnet,
	title={Proactive Cyber Defense: A Multi-Modal Fusion Framework for Predictive Threat Intelligence and Dynamic Risk Assessment},
	author={Gowda, Nagarjun K N},
	journal={arXiv preprint arXiv:2024.XXXXX},
	year={2024}
}
```

### License

MIT License — see LICENSE for details.

### Contact

Nagarjun Gowda K N — nagarjun@gmail.com

Project Link: https://github.com/nagarjungowdakn13/CyberShieldNet

## Running the API

Use Uvicorn to run the application:

```bash
uvicorn src.cybershieldnet.api.fastapi:app --host 0.0.0.0 --port 8000
```

### Run Modes

#### Python 3.14 API-only mode (Windows)

If you face dependency issues on Python 3.14 (NumPy/torch wheels or FastAPI/Pydantic mismatches), run the lightweight Starlette server:

```cmd
cd /d "C:\Users\nagar\OneDrive\Desktop\Projects ^& Research\CyberShieldNet"
"C:/Users/nagar/OneDrive/Desktop/Projects ^& Research/CyberShieldNet/.venv/Scripts/python.exe" -m pip install uvicorn starlette
"C:/Users/nagar/OneDrive/Desktop/Projects ^& Research/CyberShieldNet/.venv/Scripts/python.exe" -m uvicorn tools.starlette_server:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health` → returns `ok`
- `GET /` → basic app info
- `POST /predict` → dummy risk score, echoes input

#### Full ML mode (Python 3.10)

Use Python 3.10 with prebuilt wheels and install ML stack:

```cmd
cd /d "C:\Users\nagar\OneDrive\Desktop\Projects & Research\CyberShieldNet"
".venv\Scripts\python.exe" -m pip install -r ml-requirements.txt
".venv\Scripts\python.exe" -m uvicorn src.cybershieldnet.api.fastapi:app --host 127.0.0.1 --port 8000
```

Metrics: visit `http://127.0.0.1:8000/metrics`. Version: `http://127.0.0.1:8000/version`.

### `requirements.txt`

```txt
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
networkx>=2.6.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
shap>=0.41.0
captum>=0.5.0
fastapi>=0.95.0
uvicorn>=0.21.0
pydantic>=1.10.0
python-multipart>=0.0.6
redis>=4.5.0
celery>=5.3.0
prometheus-client>=0.17.0
grafana-api>=1.0.3
elasticsearch>=8.7.0
kafka-python>=2.0.0
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0
jupyter>=1.0.0
ipywidgets>=8.0.0
tqdm>=4.64.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

Feature flags:

- Set `CSN_ENABLE_COMPLEX=1` to enable advanced endpoints (requires full dependencies).
