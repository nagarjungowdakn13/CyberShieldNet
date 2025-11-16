"""Experiment: training runner (template)

Small CLI wrapper that delegates to the project's training helper.
This is a template for experiment automation and can be extended to
log hyperparameters, push artifacts, or integrate with experiment
tracking backends.
"""

import sys
import argparse
import logging

# Ensure local `src` package is importable when running from repo root
sys.path.insert(0, 'src')

from scripts.train_model import run_training

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Run training experiment (template)')
    parser.add_argument('--config-dir', default='config', help='Config directory')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--out', default='models/experiment_model.pth', help='Output model path')

    args = parser.parse_args()

    model_path = run_training(config_dir=args.config_dir, epochs=args.epochs, model_out=args.out)
    print(f"Experiment training complete. Model saved to: {model_path}")


if __name__ == '__main__':
    main()
