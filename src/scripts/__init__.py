"""Convenience exports for CLI scripts in the project.

This module exposes simple entrypoints that can be used by tooling
or invoked from the command line. The actual implementations are
provided in the sibling modules: `data_pipeline`, `train_model`,
`evaluate_model` and `deploy_model`.
"""

from .data_pipeline import main as data_pipeline_main
from .train_model import main as train_main
from .evaluate_model import main as evaluate_main
from .deploy_model import main as deploy_main

__all__ = [
	"data_pipeline_main",
	"train_main",
	"evaluate_main",
	"deploy_main",
]
