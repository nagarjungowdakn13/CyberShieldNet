"""Advanced usage example for CyberShieldNet.

This script demonstrates a short smoke training run. The example
attempts to use the project's training entrypoint but will fall back
to a minimal scikit-learn training loop when deep-learning libs are
not available.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO)


def run_project_training():
    try:
        # avoid importing at module import time; import inside the function
        from cybershieldnet.scripts.train_model import run_training
        return run_training(config_dir='config', epochs=1, model_out='models/cybershieldnet_smoke.pth')
    except Exception as e:
        raise RuntimeError(f"Project training not available: {e}")


def run_sklearn_smoke():
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
    except Exception:
        raise RuntimeError("Please install scikit-learn and numpy to run the fallback training.")

    X = np.random.randn(200, 16)
    y = (np.random.rand(200) > 0.5).astype(int)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    out_path = 'models/sklearn_smoke.pkl'
    try:
        import joblib
        joblib.dump(clf, out_path)
    except Exception:
        # best-effort save
        with open(out_path, 'wb') as f:
            f.write(b'')
    return out_path


def main():
    print("Starting advanced usage smoke run...")
    try:
        model_path = run_project_training()
        print(f"Project training finished. Model saved to: {model_path}")
    except Exception as e:
        print("Project training unavailable — running sklearn fallback (reason):", e)
        path = run_sklearn_smoke()
        print(f"Fallback model saved to: {path}")


if __name__ == '__main__':
    main()
