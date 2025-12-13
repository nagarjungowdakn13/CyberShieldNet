"""Basic usage example for CyberShieldNet.

This script is written to be robust when heavy dependencies (like
PyTorch) are not installed. It demonstrates the intended usage and
provides safe fallbacks using scikit-learn for environments without
`torch`.
"""

from typing import Tuple, Any
import sys
import os


def make_dummy_data(batch_size: int = 8) -> Tuple[Any, Any]:
	"""Create small dummy dataset suitable for a quick smoke demo.

	Returns (X, y) where X is a 2D array-like and y is labels.
	"""
	try:
		import numpy as _np
	except Exception:
		raise RuntimeError("Please install numpy to run the basic example.")

	X = _np.random.randn(batch_size, 32)
	y = (_np.random.rand(batch_size) > 0.5).astype(int)
	return X, y


def run_with_torch():
	"""Demonstrate a tiny forward pass using PyTorch if available."""
	try:
		import torch
		from cybershieldnet.core.base_model import CyberShieldNet
	except Exception as exc:
		raise RuntimeError("torch or cybershieldnet not importable: %s" % exc)

	model = CyberShieldNet()
	model.eval()

	# create dummy tensor matching a plausible input shape
	x = torch.randn(2, 32)
	try:
		with torch.no_grad():
			out = model.forward_from_features(x)
		print("Torch forward OK — output shape:", getattr(out, 'shape', type(out)))
	except Exception as e:
		print("Model forward raised (expected in lightweight demo):", e)


def run_with_sklearn():
	"""Fallback demo using scikit-learn to show an end-to-end predict flow."""
	try:
		from sklearn.ensemble import RandomForestClassifier
	except Exception:
		raise RuntimeError("Please install scikit-learn to run the fallback demo.")

	X, y = make_dummy_data(batch_size=64)
	clf = RandomForestClassifier(n_estimators=10)
	clf.fit(X, y)
	preds = clf.predict_proba(X[:5])[:, 1]
	print("SKLearn fallback predictions:", preds)


def main():
	# prefer torch path if available, otherwise run sklearn fallback
	try:
		run_with_torch()
	except Exception as e:
		print("Torch path unavailable — running sklearn fallback (reason):", e)
		run_with_sklearn()


if __name__ == "__main__":
	main()

