import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CyberThreatDataset(Dataset):
    """
    Dataset class for cyber threat intelligence data
    """
    
    def __init__(self, 
                 graph_data: Dict,
                 temporal_data: torch.Tensor,
                 behavioral_data: torch.Tensor,
                 labels: torch.Tensor,
                 metadata: Optional[Dict] = None):
        self.graph_data = graph_data
        self.temporal_data = temporal_data
        self.behavioral_data = behavioral_data
        self.labels = labels
        self.metadata = metadata or {}
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'graph_data': {
                'x': self.graph_data['x'][idx],
                'edge_index': self.graph_data['edge_index'],
                'edge_attr': self.graph_data['edge_attr'][idx] if 'edge_attr' in self.graph_data else None
            },
            'temporal_data': self.temporal_data[idx],
            'behavioral_data': self.behavioral_data[idx],
            'labels': self.labels[idx],
            'metadata': {k: v[idx] for k, v in self.metadata.items()}
        }

class DataLoaderFactory:
    """
    Factory class for creating data loaders for different data modalities
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.batch_size = config.get('batch_size', 32)
        
    def create_graph_loader(self, graph_data: Dict) -> DataLoader:
        """Create data loader for graph data"""
        dataset = GraphDataset(graph_data)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def create_temporal_loader(self, temporal_data: torch.Tensor) -> DataLoader:
        """Create data loader for temporal data"""
        dataset = TemporalDataset(temporal_data)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def create_multi_modal_loader(self, 
                                 graph_data: Dict,
                                 temporal_data: torch.Tensor,
                                 behavioral_data: torch.Tensor,
                                 labels: torch.Tensor) -> DataLoader:
        """Create data loader for multi-modal data"""
        dataset = CyberThreatDataset(
            graph_data, temporal_data, behavioral_data, labels
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for multi-modal data"""
        collated = {}
        
        # Collate graph data
        graph_data = {
            'x': torch.stack([item['graph_data']['x'] for item in batch]),
            'edge_index': batch[0]['graph_data']['edge_index'],  # Same for all
        }
        if batch[0]['graph_data']['edge_attr'] is not None:
            graph_data['edge_attr'] = torch.stack(
                [item['graph_data']['edge_attr'] for item in batch]
            )
        collated['graph_data'] = graph_data
        
        # Collate other data
        collated['temporal_data'] = torch.stack([item['temporal_data'] for item in batch])
        collated['behavioral_data'] = torch.stack([item['behavioral_data'] for item in batch])
        collated['labels'] = torch.stack([item['labels'] for item in batch])
        
        return collated

class GraphDataset(Dataset):
    """Dataset for graph data"""
    
    def __init__(self, graph_data: Dict):
        self.graph_data = graph_data
        
    def __len__(self):
        return self.graph_data['x'].size(0)
    
    def __getitem__(self, idx):
        return {
            'x': self.graph_data['x'][idx],
            'edge_index': self.graph_data['edge_index'],
            'edge_attr': self.graph_data['edge_attr'][idx] if 'edge_attr' in self.graph_data else None
        }

class TemporalDataset(Dataset):
    """Dataset for temporal data"""
    
    def __init__(self, temporal_data: torch.Tensor):
        self.temporal_data = temporal_data
        
    def __len__(self):
        return len(self.temporal_data)
    
    def __getitem__(self, idx):
        return self.temporal_data[idx]