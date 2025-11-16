import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class GraphConvolution(nn.Module):
    """
    Graph Convolutional Network implementation for cyber threat graphs
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass for graph convolution"""
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output

class GCNLayer(nn.Module):
    """Complete GCN layer with normalization and activation"""
    
    def __init__(self, in_features: int, out_features: int, 
                 dropout: float = 0.5, activation: str = 'relu'):
        super().__init__()
        
        self.gc = GraphConvolution(in_features, out_features)
        self.dropout = dropout
        self.activation = self._get_activation(activation)
        self.batch_norm = nn.BatchNorm1d(out_features)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.gc(x, adj)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class MultiScaleGCN(nn.Module):
    """Multi-scale graph convolutional network"""
    
    def __init__(self, in_features: int, hidden_dims: list, num_scales: int = 3):
        super().__init__()
        
        self.num_scales = num_scales
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNLayer(in_features, hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(GCNLayer(hidden_dims[i-1], hidden_dims[i]))
        
        # Multi-scale processing
        self.scale_layers = nn.ModuleList([
            GCNLayer(hidden_dims[-1], hidden_dims[-1]) for _ in range(num_scales)
        ])
        
        self.attention = ScaleAttention(hidden_dims[-1], num_scales)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale processing"""
        # Base GCN layers
        for layer in self.layers:
            x = layer(x, adj)
        
        # Multi-scale representations
        scale_outputs = []
        current_x = x
        
        for scale_layer in self.scale_layers:
            current_x = scale_layer(current_x, adj)
            scale_outputs.append(current_x)
        
        # Attention-based scale fusion
        x = self.attention(scale_outputs)
        
        return x

class ScaleAttention(nn.Module):
    """Attention mechanism for multi-scale feature fusion"""
    
    def __init__(self, feature_dim: int, num_scales: int):
        super().__init__()
        
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.softmax = nn.Softmax(dim=0)
        self.feature_transform = nn.Linear(feature_dim * num_scales, feature_dim)
    
    def forward(self, scale_outputs: list) -> torch.Tensor:
        """Fuse multi-scale features with attention"""
        # Normalize scale weights
        weights = self.softmax(self.scale_weights)
        
        # Weighted combination
        weighted_outputs = []
        for i, output in enumerate(scale_outputs):
            weighted_outputs.append(weights[i] * output)
        
        # Concatenate and transform
        fused = torch.cat(weighted_outputs, dim=-1)
        fused = self.feature_transform(fused)
        
        return fused

class GraphSAGELayer(nn.Module):
    """GraphSAGE layer for inductive graph learning"""
    
    def __init__(self, in_features: int, out_features: int, aggregator: str = 'mean'):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        self.linear_self = nn.Linear(in_features, out_features, bias=False)
        self.linear_neigh = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.linear_self.weight)
        nn.init.xavier_uniform_(self.linear_neigh.weight)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass for GraphSAGE"""
        # Self features
        x_self = self.linear_self(x)
        
        # Neighborhood aggregation
        if self.aggregator == 'mean':
            x_neigh = torch.spmm(adj, x)
        elif self.aggregator == 'max':
            # Simplified max pooling (for demonstration)
            x_neigh = torch.spmm(adj, x)
        else:
            x_neigh = torch.spmm(adj, x)
        
        x_neigh = self.linear_neigh(x_neigh)
        
        # Combine self and neighborhood
        output = x_self + x_neigh
        output = F.normalize(output, p=2, dim=1)  # L2 normalization
        
        return output

class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, alpha: float = 0.2):
        super().__init__()
        
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass for GAT layer"""
        h = torch.mm(x, self.W)
        N = h.size(0)
        
        # Self-attention mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        a_input = a_input.view(N, -1, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Output features
        h_prime = torch.matmul(attention, h)
        
        return h_prime

class DynamicGraphConv(nn.Module):
    """Dynamic graph convolution for evolving threat graphs"""
    
    def __init__(self, in_features: int, out_features: int, temporal_window: int = 5):
        super().__init__()
        
        self.temporal_window = temporal_window
        self.conv_layers = nn.ModuleList([
            GCNLayer(in_features, out_features) for _ in range(temporal_window)
        ])
        
        self.temporal_attention = nn.MultiheadAttention(out_features, num_heads=4)
        self.gru = nn.GRU(out_features, out_features, batch_first=True)
    
    def forward(self, x_sequence: list, adj_sequence: list) -> torch.Tensor:
        """Process temporal sequence of graphs"""
        temporal_features = []
        
        # Process each time step
        for t, (x, adj) in enumerate(zip(x_sequence, adj_sequence)):
            if t < len(self.conv_layers):
                features = self.conv_layers[t](x, adj)
            else:
                features = self.conv_layers[-1](x, adj)
            temporal_features.append(features.unsqueeze(1))
        
        # Stack temporal features
        temporal_stack = torch.cat(temporal_features, dim=1)
        
        # Temporal attention
        attended_features, _ = self.temporal_attention(
            temporal_stack, temporal_stack, temporal_stack
        )
        
        # GRU for temporal modeling
        gru_out, _ = self.gru(attended_features)
        
        # Use final hidden state
        output = gru_out[:, -1, :]
        
        return output