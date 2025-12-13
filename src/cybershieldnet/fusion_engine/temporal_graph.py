import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class TemporalGraphNetwork(nn.Module):
    """
    Enhanced Temporal Graph Network for cyber threat intelligence
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Network parameters
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.num_heads = config.get('num_heads', 8)
        
        # Initialize components
        self.graph_encoder = GraphEncoder(self.hidden_dims[0], self.hidden_dims, self.dropout_rate)
        self.temporal_encoder = TemporalEncoder(self.hidden_dims[0], self.hidden_dims[1], self.num_heads)
        self.cross_modal_attention = CrossModalAttention(self.hidden_dims[-1])
        self.fusion_strategy = config.get('fusion_strategy', 'concatenate')
        
        logger.info("TemporalGraphNetwork initialized")
    
    def forward(self, 
                graph_data: Dict,
                temporal_data: torch.Tensor,
                behavioral_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal graph network
        """
        # Encode graph structure
        graph_embeddings = self.graph_encoder(graph_data)
        
        # Encode temporal patterns
        temporal_embeddings = self.temporal_encoder(temporal_data)
        
        # Cross-modal attention
        fused_embeddings = self.cross_modal_attention(graph_embeddings, temporal_embeddings)
        
        # Optional behavioral data fusion
        if behavioral_data is not None:
            fused_embeddings = self._fuse_behavioral_data(fused_embeddings, behavioral_data)
        
        return fused_embeddings
    
    def _fuse_behavioral_data(self, 
                            fused_embeddings: torch.Tensor,
                            behavioral_data: torch.Tensor) -> torch.Tensor:
        """Fuse behavioral data with existing embeddings"""
        if self.fusion_strategy == 'concatenate':
            return torch.cat([fused_embeddings, behavioral_data], dim=-1)
        elif self.fusion_strategy == 'add':
            # Project behavioral data to match dimensions
            if behavioral_data.size(-1) != fused_embeddings.size(-1):
                behavioral_proj = nn.Linear(behavioral_data.size(-1), fused_embeddings.size(-1))(behavioral_data)
                return fused_embeddings + behavioral_proj
            return fused_embeddings + behavioral_data
        elif self.fusion_strategy == 'weighted':
            # Learnable weighted combination
            alpha = torch.sigmoid(self.fusion_weight)
            return alpha * fused_embeddings + (1 - alpha) * behavioral_data
        else:
            return fused_embeddings

class GraphEncoder(nn.Module):
    """Graph structure encoder with multi-head attention"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Multi-layer graph processing
        self.graph_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.graph_layers.append(
                GraphProcessingLayer(hidden_dims[i], hidden_dims[i + 1], dropout_rate)
            )
        
        self.attention_pooling = AttentionPooling(hidden_dims[-1])
    
    def forward(self, graph_data: Dict) -> torch.Tensor:
        """Encode graph structure"""
        x = graph_data['x']
        edge_index = graph_data['edge_index']
        edge_attr = graph_data.get('edge_attr', None)
        
        # Initial projection
        x = self.input_projection(x)
        
        # Graph processing layers
        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_attr)
        
        # Global graph representation
        graph_embedding = self.attention_pooling(x)
        
        return graph_embedding

class GraphProcessingLayer(nn.Module):
    """Advanced graph processing layer"""
    
    def __init__(self, in_features: int, out_features: int, dropout_rate: float):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.attention = MultiHeadGraphAttention(out_features, 4)  # 4 attention heads
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features * 2, out_features)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        """Graph processing with residual connections"""
        # Self-attention on graph nodes
        residual = x
        x = self.linear(x)
        x = self.attention(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # Feed-forward network
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        
        return x

class MultiHeadGraphAttention(nn.Module):
    """Multi-head graph attention mechanism"""
    
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Multi-head attention with graph structure"""
        batch_size, num_nodes, _ = x.shape
        
        # Project to query, key, value
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply graph structure mask
        adj_mask = self._create_adjacency_mask(edge_index, batch_size, num_nodes)
        attn_scores = attn_scores.masked_fill(adj_mask.unsqueeze(1) == 0, float('-inf'))
        
        # Softmax and attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Combine heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.feature_dim)
        output = self.out_proj(attn_output)
        
        return output
    
    def _create_adjacency_mask(self, edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        """Create adjacency mask from edge index"""
        # This is a simplified implementation
        # In practice, you might want to use a more sophisticated approach
        adj_mask = torch.ones(batch_size, num_nodes, num_nodes, device=edge_index.device)
        return adj_mask

class TemporalEncoder(nn.Module):
    """Advanced temporal sequence encoder"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.temporal_attention = TemporalMultiHeadAttention(hidden_dim, num_heads)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)  # Bidirectional
        self.dropout = nn.Dropout(0.1)
        
        # Projection to match dimensions
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """Encode temporal sequences"""
        # Project input
        x = self.input_proj(temporal_data)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Temporal self-attention
        residual = x
        x = self.temporal_attention(x)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm2(lstm_out)
        
        # Global temporal representation (mean pooling)
        temporal_embedding = torch.mean(lstm_out, dim=1)
        temporal_embedding = self.output_proj(temporal_embedding)
        
        return temporal_embedding

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x

class TemporalMultiHeadAttention(nn.Module):
    """Multi-head attention for temporal sequences"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Temporal multi-head attention"""
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask for autoregressive modeling
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(attn_output)
        
        return output

class CrossModalAttention(nn.Module):
    """Cross-modal attention between graph and temporal embeddings"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.graph_to_temp_attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        self.temp_to_graph_attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, graph_embeddings: torch.Tensor, temporal_embeddings: torch.Tensor) -> torch.Tensor:
        """Cross-modal attention fusion"""
        # Graph to temporal attention
        temp_enhanced, _ = self.graph_to_temp_attention(
            temporal_embeddings.unsqueeze(1), 
            graph_embeddings.unsqueeze(1), 
            graph_embeddings.unsqueeze(1)
        )
        temp_enhanced = temp_enhanced.squeeze(1)
        
        # Temporal to graph attention
        graph_enhanced, _ = self.temp_to_graph_attention(
            graph_embeddings.unsqueeze(1),
            temporal_embeddings.unsqueeze(1),
            temporal_embeddings.unsqueeze(1)
        )
        graph_enhanced = graph_enhanced.squeeze(1)
        
        # Gated fusion
        combined = torch.cat([graph_enhanced, temp_enhanced], dim=-1)
        fusion_gate = self.fusion_gate(combined)
        
        fused_embeddings = fusion_gate * graph_enhanced + (1 - fusion_gate) * temp_enhanced
        fused_embeddings = self.layer_norm(fused_embeddings)
        
        return fused_embeddings

class AttentionPooling(nn.Module):
    """Attention-based pooling for graph-level representations"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Attention-based graph pooling"""
        attention_weights = self.attention(node_embeddings)
        graph_embedding = torch.sum(attention_weights * node_embeddings, dim=1)
        return graph_embedding