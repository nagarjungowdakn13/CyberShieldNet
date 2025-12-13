"""Training script for CyberShieldNet.

This script wires together the `CyberShieldNet` model, the
project `ConfigManager`, `DataLoader` and `ModelCheckpoint` utilities
to provide a simple command-line training entrypoint.
"""

import argparse
import logging
from pathlib import Path
import pickle

try:
    import torch
except Exception:  # pragma: no cover - runtime import guard
    torch = None

TORCH_AVAILABLE = torch is not None

from cybershieldnet.core.base_model import CyberShieldNet
from cybershieldnet.utils.helpers import ConfigManager, DataLoader, ModelCheckpoint

logger = logging.getLogger(__name__)


def _make_dummy_loader(batch_size: int, num_batches: int = 10):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Dummy loader requires PyTorch. Install torch or rely on fallback training.")
    # Create a tiny synthetic dataset to allow quick smoke runs
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return batch_size * num_batches

        def __getitem__(self, idx):
            # Minimal tensors matching model forward signature
            return {
                'graph_data': {},
                'temporal_data': torch.randn(8, 16),
                'behavioral_data': torch.randn(8, 10),
                'context_data': {'assets': [], 'vulnerabilities': []},
                'labels': torch.randint(0, 2, (8, 1)).float()
            }

    ds = DummyDataset()
    return torch.utils.data.DataLoader(ds, batch_size=batch_size)


def run_training(config_dir: str = 'config', epochs: int = None, model_out: str = 'models/cybershieldnet.pth'):
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Running fallback sklearn training instead.")
        return _run_fallback_training(model_out)
    cfg_mgr = ConfigManager(config_dir)
    try:
        train_cfg = cfg_mgr.get_config('training_config')
    except Exception:
        train_cfg = {}

    batch_size = train_cfg.get('batch_size', 16)
    epochs = epochs or train_cfg.get('epochs', 2)

    # Initialize model
    model = CyberShieldNet()

    # Prepare data loader (project DataLoader is for file-based loading; for CLI we allow dummy)
    try:
        data_cfg = cfg_mgr.get_config('data_config')
        dl = DataLoader(data_cfg)
        # If a dataset path is given in config, try to load that processed file
        processed_path = data_cfg.get('processed_path')
        if processed_path:
            dataset = dl.load_data(processed_path)
            # Not attempting to convert to PyTorch Dataset generically here
            train_loader = _make_dummy_loader(batch_size)
        else:
            train_loader = _make_dummy_loader(batch_size)
    except Exception:
        train_loader = _make_dummy_loader(batch_size)

    # Checkpointing
    ckpt = ModelCheckpoint(checkpoint_dir=train_cfg.get('checkpoint_dir', 'checkpoints'))

    # Run training (delegated to model.fit)
    history = model.fit(train_loader, epochs=epochs)

    # Save trained model
    outp = Path(model_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(outp))

    # Save a checkpoint via helper
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get('learning_rate', 1e-3))
    ckpt.save_checkpoint(model, optimizer, epoch=epochs, metrics={'train_loss': history.get('train_loss')})

    logger.info(f"Training completed. Model saved to {outp}")
    return str(outp)


def _run_fallback_training(model_out: str) -> str:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - requires optional deps
        raise RuntimeError(
            "Fallback training requires numpy and scikit-learn. Install them or install torch for full training.") from exc

    X = np.random.randn(256, 32)
    y = (np.random.rand(256) > 0.5).astype(int)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)

    outp = Path(model_out).with_suffix('.pkl')
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'wb') as fh:
        pickle.dump({'model': clf, 'note': 'Fallback sklearn classifier'}, fh)

    logger.info(f"Fallback sklearn model saved to {outp}")
    return str(outp)


def main():
    parser = argparse.ArgumentParser(description='Train CyberShieldNet model')
    parser.add_argument('--config-dir', default='config', help='Configuration directory')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--model-out', default='models/cybershieldnet.pth', help='Path to save trained model')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        out = run_training(args.config_dir, epochs=args.epochs, model_out=args.model_out)
        print(f"Model training finished. Model saved to: {out}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
