"""
Constants and configuration for CyberShieldNet
"""

from enum import Enum
from typing import Dict, Any

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class DataModality(Enum):
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    CONTEXTUAL = "contextual"
    THREAT_INTEL = "threat_intelligence"
    ASSET_CRITICALITY = "asset_criticality"

class ModelType(Enum):
    TGCF = "temporal_graph_convolutional_fusion"
    DRPA = "dynamic_risk_propagation"
    ENSEMBLE = "adaptive_ensemble"
    LSTM = "lstm"
    GCN = "graph_convolutional_network"

# Default configuration from the paper
DEFAULT_CONFIG = {
    "tgcf": {
        "graph_layers": 3,
        "hidden_dims": [256, 128, 64],
        "lstm_hidden_size": 128,
        "lstm_layers": 2,
        "dropout_rate": 0.3,
        "activation": "relu"
    },
    "drpa": {
        "risk_decay_factor": 0.85,
        "similarity_threshold": 0.7,
        "propagation_depth": 5,
        "context_weight": 0.15
    },
    "training": {
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "batch_size": 512,
        "epochs": 200,
        "weight_decay": 0.01,
        "early_stopping_patience": 25
    },
    "ensemble": {
        "base_models": 7,
        "adaptation_rate": 0.1,
        "update_frequency": 3600  # 1 hour
    }
}

# Performance targets from the paper
PERFORMANCE_TARGETS = {
    "accuracy": 0.962,
    "precision": 0.951,
    "recall": 0.958,
    "f1_score": 0.954,
    "auc_roc": 0.989,
    "false_positive_rate": 0.083
}