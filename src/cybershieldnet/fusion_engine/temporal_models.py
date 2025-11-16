import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import logging
import math

logger = logging.getLogger(__name__)

class TemporalModel(nn.Module):
    """
    Advanced temporal modeling for cyber threat sequences
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        self.input_size = config.get('input_size', 64)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.3)
        self.bidirectional = config.get('bidirectional', True)
        
        # Model components
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        self.attention = TemporalAttention(
            self.hidden_size * (2 if self.bidirectional else 1)
        )
        
        self.conv1d = TemporalConv1D(
            self.input_size, 
            self.hidden_size,
            kernel_sizes=[3, 5, 7]
        )
        
        self.fusion_layer = TemporalFusion(
            self.hidden_size * (2 if self.bidirectional else 1) * 3  # LSTM, GRU, Conv1D
        )
        
        logger.info("TemporalModel initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal sequences with multiple models
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Temporal embeddings of shape (batch_size, hidden_size)
        """
        # LSTM processing
        lstm_out, (lstm_hidden, _) = self.lstm(x)
        lstm_features = self.attention(lstm_out)
        
        # GRU processing
        gru_out, gru_hidden = self.gru(x)
        gru_features = self.attention(gru_out)
        
        # 1D Convolution processing
        conv_features = self.conv1d(x)
        
        # Feature fusion
        combined_features = torch.cat([lstm_features, gru_features, conv_features], dim=-1)
        temporal_embedding = self.fusion_layer(combined_features)
        
        return temporal_embedding

class TemporalAttention(nn.Module):
    """Attention mechanism for temporal sequences"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to temporal sequence
        
        Args:
            hidden_states: Tensor of shape (batch_size, sequence_length, hidden_size)
            
        Returns:
            Weighted sequence representation of shape (batch_size, hidden_size)
        """
        # Compute attention scores
        attention_scores = self.attention_weights(hidden_states).squeeze(-1)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention weights
        weighted_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        
        return weighted_output

class MultiHeadTemporalAttention(nn.Module):
    """Multi-head attention for temporal sequences"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-head temporal attention
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Attended sequence (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_linear(attn_output)
        
        return output

class TemporalConv1D(nn.Module):
    """1D Convolutional network for temporal patterns"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) // 2
            conv = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=padding
            )
            pool = nn.AdaptiveMaxPool1d(1)
            
            self.conv_layers.append(conv)
            self.pool_layers.append(pool)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal sequences with 1D convolutions
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Convolutional features (batch_size, out_channels * num_kernels)
        """
        # Rearrange for conv1d: (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        conv_outputs = []
        
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            # Apply convolution
            conv_out = conv(x)
            conv_out = self.activation(conv_out)
            conv_out = self.dropout(conv_out)
            
            # Global pooling
            pooled = pool(conv_out).squeeze(-1)
            conv_outputs.append(pooled)
        
        # Concatenate features from different kernel sizes
        combined = torch.cat(conv_outputs, dim=-1)
        
        return combined

class TemporalFusion(nn.Module):
    """Fusion layer for multiple temporal representations"""
    
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 128):
        super().__init__()
        
        self.fusion_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.layer_norm = nn.LayerNorm(output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse multiple temporal representations"""
        fused = self.fusion_net(x)
        fused = self.layer_norm(fused)
        return fused

class TransformerTemporalModel(nn.Module):
    """Transformer-based temporal model"""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transformer-based temporal processing
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional sequence mask
            
        Returns:
            Encoded sequence (batch_size, seq_len, d_model)
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, mask=mask)
        
        # Global average pooling
        pooled = torch.mean(encoded, dim=1)
        output = self.output_projection(pooled)
        
        return output

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :]
        return x

class TemporalAutoencoder(nn.Module):
    """Temporal autoencoder for anomaly detection in sequences"""
    
    def __init__(self, input_size: int, hidden_size: int, sequence_length: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size * sequence_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size * sequence_length)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoencoder forward pass"""
        batch_size, seq_len, features = x.shape
        
        # Flatten sequence
        x_flat = x.view(batch_size, -1)
        
        # Encode
        encoded = self.encoder(x_flat)
        
        # Decode
        decoded = self.decoder(encoded)
        decoded = decoded.view(batch_size, seq_len, features)
        
        return encoded, decoded
    
    def compute_reconstruction_loss(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss for anomaly detection"""
        return F.mse_loss(x_reconstructed, x, reduction='mean')