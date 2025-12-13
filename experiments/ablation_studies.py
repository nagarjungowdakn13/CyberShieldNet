"""Experiment: ablation studies template

Runs a few short experiments toggling components to evaluate
their effect. This template runs small smoke training runs with
different config overrides and records where artifacts are written.
"""

import sys
import logging

sys.path.insert(0, 'src')

from scripts.train_model import run_training

logging.basicConfig(level=logging.INFO)


def run_ablation_trials():
    trials = [
        {'name': 'baseline', 'epochs': 1},
        {'name': 'shorter', 'epochs': 1},
    ]

    results = {}
    for t in trials:
        out = run_training(epochs=t['epochs'], model_out=f"models/ablation_{t['name']}.pth")
        results[t['name']] = out

    return results


def main():
    results = run_ablation_trials()
    print('Ablation results:')
    for k, v in results.items():
        print(f' - {k}: {v}')


if __name__ == '__main__':
    main()
