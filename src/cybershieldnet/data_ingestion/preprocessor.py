import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing pipeline for multi-modal threat intelligence data
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.imputers = {}
        
    def preprocess_graph_data(self, graph_data: Dict) -> Dict:
        """Preprocess graph structure data"""
        logger.info("Preprocessing graph data...")
        
        # Normalize node features
        if 'x' in graph_data and graph_data['x'] is not None:
            graph_data['x'] = self._normalize_features(graph_data['x'], 'graph_nodes')
        
        # Normalize edge features if present
        if 'edge_attr' in graph_data and graph_data['edge_attr'] is not None:
            graph_data['edge_attr'] = self._normalize_features(graph_data['edge_attr'], 'graph_edges')
        
        return graph_data
    
    def preprocess_temporal_data(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """Preprocess temporal sequence data"""
        logger.info("Preprocessing temporal data...")
        
        # Handle missing values
        temporal_data = self._impute_missing(temporal_data, 'temporal')
        
        # Normalize features
        temporal_data = self._normalize_features(temporal_data, 'temporal')
        
        return temporal_data
    
    def preprocess_behavioral_data(self, behavioral_data: torch.Tensor) -> torch.Tensor:
        """Preprocess behavioral feature data"""
        logger.info("Preprocessing behavioral data...")
        
        # Handle missing values
        behavioral_data = self._impute_missing(behavioral_data, 'behavioral')
        
        # Normalize features
        behavioral_data = self._normalize_features(behavioral_data, 'behavioral')
        
        return behavioral_data
    
    def preprocess_contextual_data(self, context_data: Dict) -> Dict:
        """Preprocess contextual information"""
        logger.info("Preprocessing contextual data...")
        
        processed_context = {}
        
        for key, value in context_data.items():
            if isinstance(value, torch.Tensor):
                # Handle tensor data
                value = self._impute_missing(value, f'context_{key}')
                value = self._normalize_features(value, f'context_{key}')
                processed_context[key] = value
            elif isinstance(value, (list, np.ndarray)):
                # Handle array data
                value_tensor = torch.tensor(value, dtype=torch.float32)
                value_tensor = self._impute_missing(value_tensor, f'context_{key}')
                value_tensor = self._normalize_features(value_tensor, f'context_{key}')
                processed_context[key] = value_tensor
            else:
                # Keep other data types as-is
                processed_context[key] = value
        
        return processed_context
    
    def _normalize_features(self, features: torch.Tensor, feature_type: str) -> torch.Tensor:
        """Normalize features using appropriate scaler"""
        if feature_type not in self.scalers:
            normalization = self.config.get('preprocessing', {}).get('normalization', 'standard')
            
            if normalization == 'standard':
                self.scalers[feature_type] = StandardScaler()
            elif normalization == 'minmax':
                self.scalers[feature_type] = MinMaxScaler()
            else:
                # No normalization
                return features
        
        # Convert to numpy for sklearn
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        
        # Handle 1D arrays
        if features_np.ndim == 1:
            features_np = features_np.reshape(-1, 1)
        
        # Fit and transform
        if hasattr(self.scalers[feature_type], 'fit'):
            # Training phase - fit the scaler
            features_normalized = self.scalers[feature_type].fit_transform(features_np)
        else:
            # Inference phase - transform only
            features_normalized = self.scalers[feature_type].transform(features_np)
        
        # Convert back to tensor
        return torch.tensor(features_normalized, dtype=torch.float32)
    
    def _impute_missing(self, data: torch.Tensor, data_type: str) -> torch.Tensor:
        """Handle missing values in data"""
        strategy = self.config.get('preprocessing', {}).get('handle_missing', 'mean')
        
        if data_type not in self.imputers:
            self.imputers[data_type] = SimpleImputer(strategy=strategy)
        
        # Convert to numpy for sklearn
        data_np = data.numpy() if isinstance(data, torch.Tensor) else data
        
        # Handle 1D arrays
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)
        
        # Fit and transform
        if hasattr(self.imputers[data_type], 'fit'):
            # Training phase - fit the imputer
            data_imputed = self.imputers[data_type].fit_transform(data_np)
        else:
            # Inference phase - transform only
            data_imputed = self.imputers[data_type].transform(data_np)
        
        # Convert back to tensor
        return torch.tensor(data_imputed, dtype=torch.float32)
    
    def save_preprocessors(self, path: str):
        """Save fitted preprocessors"""
        import joblib
        
        preprocessors = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'config': self.config
        }
        joblib.dump(preprocessors, path)
        logger.info(f"Preprocessors saved to {path}")
    
    def load_preprocessors(self, path: str):
        """Load fitted preprocessors"""
        import joblib
        
        preprocessors = joblib.load(path)
        self.scalers = preprocessors['scalers']
        self.imputers = preprocessors['imputers']
        self.config = preprocessors['config']
        logger.info(f"Preprocessors loaded from {path}")