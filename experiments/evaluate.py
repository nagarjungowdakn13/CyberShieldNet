"""Experiment: evaluation runner (template)

Simple wrapper that calls the package evaluation helper. Intended
to be adapted to load real datasets and compute final metrics.
"""

import sys
import argparse
import logging

sys.path.insert(0, 'src')

from scripts.evaluate_model import run_evaluation

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Run evaluation experiment (template)')
    parser.add_argument('--model-path', required=True, help='Path to saved model')
    parser.add_argument('--config-dir', default='config', help='Config directory')
    args = parser.parse_args()

    metrics = run_evaluation(args.model_path, config_dir=args.config_dir)
    print('Evaluation metrics:')
    for k, v in metrics.items():
        print(f' - {k}: {v}')


if __name__ == '__main__':
    main()
