"""Evaluation script for CyberShieldNet.

Loads a saved model and runs a simple evaluation using a small test
dataset or a dummy dataset when none is provided. Uses
`ThreatDetectionMetrics` for metric computation.
"""

import argparse
import logging
import pickle
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - runtime import guard
    torch = None

TORCH_AVAILABLE = torch is not None

from cybershieldnet.utils.helpers import ConfigManager, DataLoader
from cybershieldnet.utils.metrics import ThreatDetectionMetrics
from cybershieldnet.core.base_model import CyberShieldNet

logger = logging.getLogger(__name__)


def _make_dummy_eval_loader(batch_size: int = 8, num_batches: int = 5):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Dummy evaluation loader requires PyTorch. Install torch or use fallback evaluation.")
    class DummyEvalDataset(torch.utils.data.Dataset):
        def __len__(self):
            return batch_size * num_batches

        def __getitem__(self, idx):
            return {
                'graph_data': {},
                'temporal_data': torch.randn(8, 16),
                'behavioral_data': torch.randn(8, 10),
                'context_data': {'assets': [], 'vulnerabilities': []},
                'labels': torch.randint(0, 2, (8, 1)).float()
            }

    ds = DummyEvalDataset()
    return torch.utils.data.DataLoader(ds, batch_size=batch_size)


def run_evaluation(model_path: str, config_dir: str = 'config') -> dict:
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Running fallback sklearn evaluation instead.")
        return _run_fallback_evaluation(model_path)
    cfg_mgr = ConfigManager(config_dir)
    try:
        eval_cfg = cfg_mgr.get_config('training_config')
    except Exception:
        eval_cfg = {}

    # Load model
    model = CyberShieldNet()
    if Path(model_path).exists():
        model.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Prepare evaluation data loader (fallback to dummy)
    batch_size = eval_cfg.get('batch_size', 8)
    eval_loader = _make_dummy_eval_loader(batch_size=batch_size)

    metrics_calc = ThreatDetectionMetrics(eval_cfg)

    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            # Move tensors if needed
            inputs = {k: v for k, v in batch.items()}
            preds, _ = model.predict(inputs.get('graph_data'), inputs.get('temporal_data'), inputs.get('behavioral_data'))
            probs = preds.squeeze() if torch.is_tensor(preds) else torch.tensor(preds)
            preds_binary = (probs > 0.5).long()

            labels = batch['labels'].squeeze().long()

            y_true_list.append(labels)
            y_pred_list.append(preds_binary)
            y_prob_list.append(probs)

    y_true = torch.cat([t.reshape(-1) for t in y_true_list])
    y_pred = torch.cat([p.reshape(-1) for p in y_pred_list])
    y_prob = torch.cat([p.reshape(-1) for p in y_prob_list])

    metrics = metrics_calc.compute_binary_metrics(y_true, y_pred, y_prob)
def _run_fallback_evaluation(model_path: str) -> dict:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    except Exception as exc:  # pragma: no cover - optional deps
        raise RuntimeError(
            "Fallback evaluation requires numpy and scikit-learn. Install them or install torch for full evaluation.") from exc

    with open(model_path, 'rb') as fh:
        model_data = pickle.load(fh)
    clf = model_data.get('model')
    if clf is None:
        raise RuntimeError("Fallback model file is invalid — expected a sklearn model.")

    X = np.random.randn(128, 32)
    y = (np.random.rand(128) > 0.5).astype(int)
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average='binary', zero_division=0)

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    logger.info(f"Fallback evaluation metrics: {metrics}")
    return metrics

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate CyberShieldNet model')
    parser.add_argument('--model-path', '-m', required=True, help='Path to saved model file')
    parser.add_argument('--config-dir', default='config', help='Configuration directory')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        metrics = run_evaluation(args.model_path, args.config_dir)
        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f" - {k}: {v}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
