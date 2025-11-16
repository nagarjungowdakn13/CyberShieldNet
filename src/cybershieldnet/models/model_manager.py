import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model management for CyberShieldNet
    Handles model saving, loading, versioning, and updates
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model_registry = {}
        self.current_models = {}
        
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)

    def register_model(self, model_name: str, model: nn.Module, metadata: Dict):
        """Register a model in the registry"""
        self.model_registry[model_name] = {
            'model': model,
            'metadata': metadata,
            'version': metadata.get('version', '1.0.0'),
            'timestamp': metadata.get('timestamp', 'unknown')
        }

    def save_model(self, model_name: str, path: Optional[str] = None):
        """Save a model to disk"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if path is None:
            path = self.model_dir / f"{model_name}_{self.model_registry[model_name]['version']}.pth"
        
        model_data = {
            'model_state_dict': self.model_registry[model_name]['model'].state_dict(),
            'metadata': self.model_registry[model_name]['metadata'],
            'version': self.model_registry[model_name]['version']
        }
        
        torch.save(model_data, path)
        logger.info(f"Model {model_name} saved to {path}")

    def load_model(self, model_name: str, path: str) -> nn.Module:
        """Load a model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found")
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # Reconstruct model architecture
        model = self._reconstruct_model(checkpoint['metadata'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Register loaded model
        self.register_model(model_name, model, checkpoint['metadata'])
        
        logger.info(f"Model {model_name} loaded from {path}")
        return model

    def _reconstruct_model(self, metadata: Dict) -> nn.Module:
        """Reconstruct model architecture from metadata"""
        # This would need to be implemented based on your model serialization strategy
        # For now, return a placeholder
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def update_model(self, model_name: str, new_model: nn.Module, update_metadata: Dict):
        """Update a model with a new version"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        old_version = self.model_registry[model_name]['version']
        new_version = self._increment_version(old_version)
        
        # Update registry
        self.model_registry[model_name] = {
            'model': new_model,
            'metadata': {**update_metadata, 'version': new_version},
            'version': new_version,
            'timestamp': update_metadata.get('timestamp', 'unknown')
        }
        
        logger.info(f"Model {model_name} updated from {old_version} to {new_version}")

    def _increment_version(self, version: str) -> str:
        """Increment model version"""
        major, minor, patch = map(int, version.split('.'))
        patch += 1
        return f"{major}.{minor}.{patch}"

    def get_model_performance(self, model_name: str) -> Dict:
        """Get performance metrics for a model"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        metadata = self.model_registry[model_name]['metadata']
        return metadata.get('performance_metrics', {})

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.model_registry.keys())

    def export_model(self, model_name: str, export_format: str = 'onnx') -> str:
        """Export model to different formats"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model = self.model_registry[model_name]['model']
        
        if export_format == 'onnx':
            return self._export_to_onnx(model, model_name)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    def _export_to_onnx(self, model: nn.Module, model_name: str) -> str:
        """Export model to ONNX format"""
        # Create dummy input
        dummy_input = torch.randn(1, 64)  # Adjust based on your model input
        
        # Export path
        export_path = self.model_dir / f"{model_name}.onnx"
        
        # Export model
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {export_path}")
        return str(export_path)

class ModelVersionControl:
    """
    Version control for model artifacts
    """

    def __init__(self, config: Dict):
        self.config = config
        self.version_history = {}

    def track_experiment(self, experiment_name: str, model_config: Dict, results: Dict):
        """Track model experiment and results"""
        if experiment_name not in self.version_history:
            self.version_history[experiment_name] = []
        
        experiment_record = {
            'timestamp': self._get_current_timestamp(),
            'config': model_config,
            'results': results,
            'version': f"1.{len(self.version_history[experiment_name]) + 1}"
        }
        
        self.version_history[experiment_name].append(experiment_record)

    def _get_current_timestamp(self) -> str:
        """Get current timestamp for versioning"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_best_model(self, experiment_name: str, metric: str = 'accuracy') -> Dict:
        """Get the best model based on a metric"""
        if experiment_name not in self.version_history:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        experiments = self.version_history[experiment_name]
        best_experiment = max(experiments, key=lambda x: x['results'].get(metric, 0))
        
        return best_experiment