"""Experiment: hyperparameter tuning template.

This file demonstrates a minimal approach to search over a few
hyperparameters. It uses Optuna if available; otherwise it falls
back to a simple grid search. The actual training call is delegated
to the `run_training` helper and should be adapted for real use.
"""

import sys
import logging
from itertools import product

sys.path.insert(0, 'src')

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

from scripts.train_model import run_training

logging.basicConfig(level=logging.INFO)


def grid_search(param_grid, trials=4):
    best = None
    for vals in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), vals))
        logging.info(f"Running trial with params: {params}")
        try:
            # For template, pass epochs via run_training; extend as needed
            run_training(epochs=params.get('epochs', 1))
            best = params
        except Exception as e:
            logging.warning(f"Trial failed: {e}")
    return best


def optuna_objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 1, 3)
    try:
        run_training(epochs=epochs)
        # Placeholder score
        return 0.0
    except Exception:
        return float('inf')


def main():
    param_grid = {'epochs': [1, 2], 'lr': [1e-4, 1e-3]}

    if HAS_OPTUNA:
        study = optuna.create_study(direction='minimize')
        study.optimize(optuna_objective, n_trials=4)
        print('Best trial:', study.best_trial.params)
    else:
        print('Optuna not installed — falling back to grid search')
        best = grid_search(param_grid)
        print('Best params (grid):', best)


if __name__ == '__main__':
    main()
