import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CyberShieldNet(nn.Module):
    """
    Main CyberShieldNet model integrating all components
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.fusion_engine = None  # Will be initialized from other modules
        self.risk_assessor = None  # Will be initialized from other modules
        self.ensemble = None  # Will be initialized from other modules
        
        # Model state
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"CyberShieldNet initialized on device: {self.device}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def forward(self, 
                graph_data: Dict,
                temporal_data: torch.Tensor,
                behavioral_data: torch.Tensor,
                context_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model
        
        Args:
            graph_data: Graph structure and features
            temporal_data: Temporal sequence data
            behavioral_data: Behavioral feature data
            context_data: Contextual information
            
        Returns:
            Tuple of (threat_predictions, risk_scores)
        """
        # Feature fusion
        fused_features = self.fusion_engine(graph_data, temporal_data, behavioral_data)
        
        # Ensemble prediction
        threat_predictions = self.ensemble(fused_features)
        
        # Risk assessment
        risk_scores = self.risk_assessor(
            threat_predictions, 
            context_data['assets'],
            context_data['vulnerabilities']
        )
        
        return threat_predictions, risk_scores
    
    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the complete model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        epochs = epochs or self.config['training']['epochs']
        self.to(self.device)
        self.train()
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        criterion = nn.BCEWithLogitsLoss()
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            history['train_loss'].append(train_loss)
            
            if val_loader:
                val_loss = self._validate_epoch(val_loader, criterion)
                history['val_loss'].append(val_loss)
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        self.is_trained = True
        return history
    
    def _train_epoch(self, data_loader, optimizer, criterion) -> float:
        """Single training epoch"""
        total_loss = 0.0
        
        for batch in data_loader:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            predictions, _ = self(**batch)
            loss = criterion(predictions, batch['labels'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def _validate_epoch(self, data_loader, criterion) -> float:
        """Single validation epoch"""
        total_loss = 0.0
        self.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                predictions, _ = self(**batch)
                loss = criterion(predictions, batch['labels'])
                total_loss += loss.item()
        
        self.train()
        return total_loss / len(data_loader)
    
    def predict(self, 
                graph_data: Dict,
                temporal_data: torch.Tensor,
                behavioral_data: torch.Tensor) -> torch.Tensor:
        """
        Make threat predictions
        
        Args:
            graph_data: Graph structure and features
            temporal_data: Temporal sequence data
            behavioral_data: Behavioral feature data
            
        Returns:
            Threat predictions
        """
        self.eval()
        with torch.no_grad():
            fused_features = self.fusion_engine(graph_data, temporal_data, behavioral_data)
            predictions = self.ensemble(fused_features)
        return torch.sigmoid(predictions)
    
    def assess_risk(self,
                   predictions: torch.Tensor,
                   assets: Dict,
                   vulnerabilities: Dict) -> torch.Tensor:
        """
        Assess organizational risk
        
        Args:
            predictions: Threat predictions
            assets: Asset information
            vulnerabilities: Vulnerability data
            
        Returns:
            Risk scores
        """
        return self.risk_assessor(predictions, assets, vulnerabilities)
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to appropriate device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in value.items()}
            else:
                device_batch[key] = value
        return device_batch
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        logger.info(f"Model loaded from {path}")