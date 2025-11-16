import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TemporalGraphFusion(nn.Module):
    """
    Temporal-Graph Convolutional Fusion (TGCF) mechanism
    Combines temporal patterns with structural dependencies
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Graph convolution parameters
        self.graph_layers = config.get('graph_layers', 3)
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        self.dropout_rate = config.get('dropout_rate', 0.3)
        
        # Temporal modeling parameters
        self.lstm_hidden_size = config.get('lstm_hidden_size', 128)
        self.lstm_layers = config.get('lstm_layers', 2)
        
        # Initialize components
        self.graph_conv = GraphConvolution(self.hidden_dims, self.dropout_rate)
        self.temporal_model = TemporalModel(self.lstm_hidden_size, self.lstm_layers, self.dropout_rate)
        self.fusion_layer = FusionLayer(self.hidden_dims[-1], self.lstm_hidden_size)
        
        logger.info("TemporalGraphFusion initialized")
    
    def forward(self, 
                graph_data: Dict,
                temporal_data: torch.Tensor,
                behavioral_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TGCF
        
        Args:
            graph_data: Graph structure and features
            temporal_data: Temporal sequence data
            behavioral_data: Behavioral feature data
            
        Returns:
            Fused feature representations
        """
        # Graph convolution
        graph_features = self.graph_conv(graph_data)
        
        # Temporal modeling
        temporal_features = self.temporal_model(temporal_data)
        
        # Feature fusion
        fused_features = self.fusion_layer(graph_features, temporal_features, behavioral_data)
        
        return fused_features

class GraphConvolution(nn.Module):
    """Graph convolutional network for structural feature extraction"""
    
    def __init__(self, hidden_dims: List[int], dropout_rate: float):
        super().__init__()
        
        self.layers = nn.ModuleList()
        input_dim = hidden_dims[0] if hidden_dims else 64
        
        # Create GCN layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                GCNLayer(hidden_dims[i], hidden_dims[i + 1], dropout_rate)
            )
        
        self.attention = GraphAttention(hidden_dims[-1] if hidden_dims else input_dim)
    
    def forward(self, graph_data: Dict) -> torch.Tensor:
        """Forward pass through graph convolution"""
        x = graph_data['x']
        edge_index = graph_data['edge_index']
        edge_attr = graph_data.get('edge_attr', None)
        
        # Apply GCN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Apply graph attention
        x = self.attention(x, edge_index)
        
        return x

class GCNLayer(nn.Module):
    """Single graph convolution layer"""
    
    def __init__(self, in_features: int, out_features: int, dropout_rate: float):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_features)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for GCN layer"""
        # Simple message passing (can be enhanced with PyTorch Geometric)
        # For simplicity, we use a basic implementation
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

class GraphAttention(nn.Module):
    """Graph attention mechanism for node importance weighting"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.attention_linear = nn.Linear(feature_dim, 1)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply graph attention"""
        # Compute attention scores
        attention_scores = self.attention_linear(x).squeeze(-1)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention weights
        x_attended = x * attention_weights.unsqueeze(-1)
        
        return x_attended

class TemporalModel(nn.Module):
    """Temporal modeling with LSTM"""
    
    def __init__(self, hidden_size: int, num_layers: int, dropout_rate: float):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # Assuming input size matches hidden size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = TemporalAttention(hidden_size * 2)  # Bidirectional
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal model"""
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(temporal_data)
        
        # Apply temporal attention
        temporal_features = self.attention(lstm_out)
        
        return self.dropout(temporal_features)

class TemporalAttention(nn.Module):
    """Temporal attention mechanism"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.attention_linear = nn.Linear(feature_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, temporal_sequences: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention"""
        # Compute attention scores for each time step
        attention_scores = self.attention_linear(temporal_sequences).squeeze(-1)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention weights
        attended_sequence = torch.sum(temporal_sequences * attention_weights.unsqueeze(-1), dim=1)
        
        return attended_sequence

class FusionLayer(nn.Module):
    """Feature fusion layer combining graph, temporal, and behavioral features"""
    
    def __init__(self, graph_dim: int, temporal_dim: int, behavioral_dim: Optional[int] = None):
        super().__init__()
        
        self.graph_projection = nn.Linear(graph_dim, 128)
        self.temporal_projection = nn.Linear(temporal_dim, 128)
        
        if behavioral_dim is not None:
            self.behavioral_projection = nn.Linear(behavioral_dim, 128)
            total_dim = 128 * 3
        else:
            self.behavioral_projection = None
            total_dim = 128 * 2
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        self.layer_norm = nn.LayerNorm(64)
    
    def forward(self, 
                graph_features: torch.Tensor,
                temporal_features: torch.Tensor,
                behavioral_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse different feature modalities"""
        # Project features to common dimension
        graph_proj = self.graph_projection(graph_features)
        temporal_proj = self.temporal_projection(temporal_features)
        
        # Combine features
        if behavioral_features is not None and self.behavioral_projection is not None:
            behavioral_proj = self.behavioral_projection(behavioral_features)
            combined = torch.cat([graph_proj, temporal_proj, behavioral_proj], dim=-1)
        else:
            combined = torch.cat([graph_proj, temporal_proj], dim=-1)
        
        # Fusion MLP
        fused_features = self.fusion_mlp(combined)
        fused_features = self.layer_norm(fused_features)
        
        return fused_features