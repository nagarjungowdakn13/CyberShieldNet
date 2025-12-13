import argparse
import json
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, average_precision_score
except Exception:
    accuracy_score = None
    precision_recall_curve = None
    roc_curve = None
    auc = None
    average_precision_score = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import torch
except Exception:
    torch = None

# Avoid importing project modules to keep this runnable in minimal env
ConfigManager = None


def load_dummy_data(n=512, features=32):
    if np is None:
        # Fallback: generate data using Python's random and simple lists
        import random
        random.seed(42)
        X = [[random.gauss(0, 1) for _ in range(features)] for _ in range(n)]
        y = [1 if random.random() > 0.5 else 0 for _ in range(n)]
        # Convert to minimal numeric arrays for downstream operations
        # Compute means for scoring later
        return X, y
    rng = np.random.RandomState(42)
    X = rng.randn(n, features)
    y = (rng.rand(n) > 0.5).astype(int)
    return X, y


def evaluate_model(config_path: str, results_out: str):
    # Config loading skipped in lite mode to avoid project imports
    model_cfg = {}

    # For demonstration purposes, use a synthetic dataset and a simple classifier
    X, y = load_dummy_data()
    # Score with a simple baseline
    if np is not None:
        X_arr = np.asarray(X)
        y_scores = np.clip(X_arr.mean(axis=1) + np.random.randn(X_arr.shape[0]) * 0.1, -3, 3)
        y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        # Pure Python scoring: mean of features and simple normalization
        import random
        means = [sum(row) / (len(row) or 1) for row in X]
        noise = [random.gauss(0, 0.1) for _ in means]
        y_scores = [max(-3, min(3, m + n)) for m, n in zip(means, noise)]
        mn, mx = min(y_scores), max(y_scores)
        denom = (mx - mn) if (mx - mn) != 0 else 1e-8
        y_prob = [(s - mn) / denom for s in y_scores]
        y_pred = [1 if p > 0.5 else 0 for p in y_prob]

    # Metrics
    if accuracy_score is not None and np is not None:
        acc = float(accuracy_score(y, y_pred))
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = float(auc(fpr, tpr))
        precision, recall, _ = precision_recall_curve(y, y_prob)
        ap = float(average_precision_score(y, y_prob))
    else:
        # Pure Python approximations
        total = len(y)
        correct = sum(1 for yt, yp in zip(y, y_pred) if int(yt) == int(yp))
        acc = correct / (total or 1)
        # Simple trapezoidal approximation for ROC using thresholds
        thresholds = [i / 20.0 for i in range(21)]
        points = []
        for th in thresholds:
            tp = fp = tn = fn = 0
            for yt, p in zip(y, y_prob):
                pred = 1 if p >= th else 0
                if yt == 1 and pred == 1:
                    tp += 1
                elif yt == 0 and pred == 1:
                    fp += 1
                elif yt == 0 and pred == 0:
                    tn += 1
                else:
                    fn += 1
            tpr = tp / (tp + fn or 1)
            fpr = fp / (fp + tn or 1)
            points.append((fpr, tpr))
        points.sort()
        roc_auc = 0.0
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            roc_auc += (x2 - x1) * (y1 + y2) / 2.0
        # Approximate AP via sorted probabilities
        pairs = sorted(zip(y_prob, y), reverse=True)
        cum_tp = 0
        cum_fp = 0
        ap_sum = 0.0
        pos = sum(1 for v in y if v == 1)
        for rank, (_, label) in enumerate(pairs, start=1):
            if label == 1:
                cum_tp += 1
            else:
                cum_fp += 1
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / (pos or 1)
            ap_sum += precision if label == 1 else 0.0
        ap = ap_sum / (pos or 1)

    out_dir = Path(results_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'average_precision': float(ap)
    }
    with open(results_out, 'w') as fh:
        json.dump(metrics, fh, indent=2)

    # Save plots only if matplotlib and sklearn curves are available
    if plt is not None and accuracy_score is not None and np is not None:
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'roc.png')

        plt.figure()
        plt.plot(recall, precision, label=f'AP={ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'pr.png')

    print(f"Saved metrics to {results_out} and plots to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/model_config.yaml')
    parser.add_argument('--results-out', default='data/results/metrics.json')
    args = parser.parse_args()

    evaluate_model(args.config, args.results_out)
