from __future__ import annotations

from typing import List, Dict, Optional
import logging

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - runtime import guard
    torch = None
    nn = None

TORCH_AVAILABLE = torch is not None and nn is not None

logger = logging.getLogger(__name__)

class AdaptiveEnsemble(nn.Module if TORCH_AVAILABLE else object):
    """
    Adaptive ensemble learning for threat prediction
    Dynamically adjusts model weights based on performance
    """

    def __init__(self, config: Dict):
        if not TORCH_AVAILABLE:
            raise RuntimeError("AdaptiveEnsemble requires PyTorch. Install torch to use this component.")
        super().__init__()
        self.config = config
        
        self.base_models = nn.ModuleList()
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.update_frequency = config.get('update_frequency', 3600)
        
        # Initialize base models
        self._initialize_base_models()
        
        # Model weights (initialized equally)
        num_models = len(self.base_models)
        self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
        self.performance_history = []
        
        logger.info(f"AdaptiveEnsemble initialized with {num_models} base models")

    def _initialize_base_models(self):
        """Initialize the base models for the ensemble"""
        # Example base models (in practice, these would be more sophisticated)
        self.base_models.append(nn.Linear(64, 1))  # Simple linear model
        self.base_models.append(nn.Sequential(     # MLP
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ))
        self.base_models.append(nn.Sequential(     # Deeper MLP
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        model_outputs = []
        
        for model in self.base_models:
            output = model(x)
            model_outputs.append(output)
        
        # Stack outputs and apply weights
        stacked_outputs = torch.stack(model_outputs, dim=-1)  # (batch_size, 1, num_models)
        weighted_output = torch.sum(stacked_outputs * self.model_weights, dim=-1)
        
        return weighted_output

    def update_weights(self, performances: List[float]):
        """Update model weights based on recent performances"""
        performances_tensor = torch.tensor(performances, dtype=torch.float32)
        
        # Softmax over performances to get new weights
        new_weights = torch.softmax(performances_tensor / self.adaptation_rate, dim=0)
        
        # Smooth update
        self.model_weights.data = (
            (1 - self.adaptation_rate) * self.model_weights + 
            self.adaptation_rate * new_weights
        )
        
        logger.info(f"Updated ensemble weights: {self.model_weights.data}")

    def evaluate_model_performance(self, 
                                 predictions: torch.Tensor, 
                                 targets: torch.Tensor) -> List[float]:
        """Evaluate performance of each base model"""
        performances = []
        
        with torch.no_grad():
            for model in self.base_models:
                model_pred = model(predictions)
                # Example performance metric (accuracy for binary classification)
                accuracy = ((model_pred > 0.5) == (targets > 0.5)).float().mean()
                performances.append(accuracy.item())
        
        return performances

class DiversityEnsemble(nn.Module if TORCH_AVAILABLE else object):
    """
    Ensemble with diversity promotion for adversarial robustness
    """

    def __init__(self, config: Dict):
        if not TORCH_AVAILABLE:
            raise RuntimeError("DiversityEnsemble requires PyTorch. Install torch to use this component.")
        super().__init__()
        self.config = config
        self.diversity_weight = config.get('diversity_weight', 0.1)
        
        self.base_models = nn.ModuleList()
        self._initialize_diverse_models()

    def _initialize_diverse_models(self):
        """Initialize diverse base models"""
        # Models with different architectures for diversity
        self.base_models.append(self._create_linear_model())
        self.base_models.append(self._create_mlp_model())
        self.base_models.append(self._create_residual_model())

    def _create_linear_model(self) -> nn.Module:
        """Create a simple linear model"""
        return nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _create_mlp_model(self) -> nn.Module:
        """Create a multi-layer perceptron"""
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def _create_residual_model(self) -> nn.Module:
        """Create a model with residual connections"""
        class ResidualBlock(nn.Module if TORCH_AVAILABLE else object):
            def __init__(self, features):
                if not TORCH_AVAILABLE:
                    raise RuntimeError("ResidualBlock requires PyTorch. Install torch to use this component.")
                super().__init__()
                self.linear1 = nn.Linear(features, features)
                self.linear2 = nn.Linear(features, features)
                self.activation = nn.ReLU()

            def forward(self, x):
                residual = x
                x = self.linear1(x)
                x = self.activation(x)
                x = self.linear2(x)
                return x + residual

        return nn.Sequential(
            nn.Linear(64, 32),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with diversity consideration"""
        model_outputs = []
        
        for model in self.base_models:
            output = model(x)
            model_outputs.append(output)
        
        # Average predictions
        stacked_outputs = torch.stack(model_outputs, dim=0)
        ensemble_output = torch.mean(stacked_outputs, dim=0)
        
        return ensemble_output

    def compute_diversity_penalty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute diversity penalty to encourage model disagreement"""
        num_models = len(self.base_models)
        diversity_penalty = 0.0
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                # Correlation between model predictions
                corr = torch.corrcoef(torch.stack([predictions[i], predictions[j]]))[0, 1]
                diversity_penalty += corr.abs()
        
        # Normalize by number of pairs
        diversity_penalty /= (num_models * (num_models - 1)) / 2
        
        return diversity_penalty