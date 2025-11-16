import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Results of data validation"""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    statistics: Dict

class DataValidator:
    """
    Data validation and quality assurance for threat intelligence data
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.quality_threshold = config.get('validation', {}).get('data_quality_threshold', 0.95)
        
    def validate_graph_data(self, graph_data: Dict) -> ValidationResult:
        """Validate graph structure data"""
        issues = []
        warnings = []
        statistics = {}
        
        # Check required fields
        required_fields = ['x', 'edge_index']
        for field in required_fields:
            if field not in graph_data:
                issues.append(f"Missing required field: {field}")
        
        if issues:
            return ValidationResult(False, issues, warnings, statistics)
        
        # Validate dimensions
        num_nodes = graph_data['x'].size(0)
        edge_index = graph_data['edge_index']
        
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            issues.append("Edge index must be 2xE tensor")
        
        # Check for self-loops and duplicates
        if edge_index.size(1) > 0:
            # Check for self-loops
            self_loops = (edge_index[0] == edge_index[1]).sum().item()
            if self_loops > 0:
                warnings.append(f"Found {self_loops} self-loops in graph")
            
            # Check for duplicate edges
            edges_set = set()
            for i in range(edge_index.size(1)):
                edge = (edge_index[0, i].item(), edge_index[1, i].item())
                if edge in edges_set:
                    warnings.append("Found duplicate edges")
                edges_set.add(edge)
        
        # Calculate statistics
        statistics.update({
            'num_nodes': num_nodes,
            'num_edges': edge_index.size(1),
            'feature_dim': graph_data['x'].size(1),
            'density': self._calculate_graph_density(num_nodes, edge_index.size(1))
        })
        
        return ValidationResult(True, issues, warnings, statistics)
    
    def validate_temporal_data(self, temporal_data: torch.Tensor) -> ValidationResult:
        """Validate temporal sequence data"""
        issues = []
        warnings = []
        statistics = {}
        
        # Check for NaN values
        nan_count = torch.isnan(temporal_data).sum().item()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values in temporal data")
        
        # Check for infinite values
        inf_count = torch.isinf(temporal_data).sum().item()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values in temporal data")
        
        # Check sequence lengths
        if temporal_data.dim() != 3:
            issues.append("Temporal data should be 3D tensor (batch, sequence, features)")
        else:
            seq_lengths = [seq.size(0) for seq in temporal_data]
            if len(set(seq_lengths)) > 1:
                warnings.append("Variable sequence lengths detected")
        
        # Calculate statistics
        if temporal_data.numel() > 0:
            statistics.update({
                'batch_size': temporal_data.size(0),
                'sequence_length': temporal_data.size(1),
                'num_features': temporal_data.size(2),
                'mean': temporal_data.mean().item(),
                'std': temporal_data.std().item(),
                'min': temporal_data.min().item(),
                'max': temporal_data.max().item()
            })
        
        is_valid = len(issues) == 0 and (nan_count + inf_count) / temporal_data.numel() < (1 - self.quality_threshold)
        
        return ValidationResult(is_valid, issues, warnings, statistics)
    
    def validate_behavioral_data(self, behavioral_data: torch.Tensor) -> ValidationResult:
        """Validate behavioral feature data"""
        issues = []
        warnings = []
        statistics = {}
        
        # Check for NaN values
        nan_count = torch.isnan(behavioral_data).sum().item()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values in behavioral data")
        
        # Check for infinite values
        inf_count = torch.isinf(behavioral_data).sum().item()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values in behavioral data")
        
        # Check for constant features
        if behavioral_data.dim() == 2:
            for i in range(behavioral_data.size(1)):
                feature = behavioral_data[:, i]
                if feature.std() == 0:
                    warnings.append(f"Feature {i} is constant")
        
        # Calculate statistics
        if behavioral_data.numel() > 0:
            statistics.update({
                'num_samples': behavioral_data.size(0),
                'num_features': behavioral_data.size(1),
                'mean': behavioral_data.mean().item(),
                'std': behavioral_data.std().item(),
                'sparsity': (behavioral_data == 0).float().mean().item()
            })
        
        is_valid = len(issues) == 0 and (nan_count + inf_count) / behavioral_data.numel() < (1 - self.quality_threshold)
        
        return ValidationResult(is_valid, issues, warnings, statistics)
    
    def validate_labels(self, labels: torch.Tensor) -> ValidationResult:
        """Validate label data"""
        issues = []
        warnings = []
        statistics = {}
        
        # Check for NaN values
        nan_count = torch.isnan(labels).sum().item()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values in labels")
        
        # Check label distribution
        if labels.dim() == 1:  # Binary classification
            unique_labels = torch.unique(labels)
            if len(unique_labels) > 2:
                warnings.append("More than 2 unique labels detected for binary classification")
            
            # Check class imbalance
            if len(unique_labels) == 2:
                class_counts = [(labels == label).sum().item() for label in unique_labels]
                imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')
                if imbalance_ratio > 10:
                    warnings.append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
                
                statistics['class_distribution'] = {
                    label.item(): count for label, count in zip(unique_labels, class_counts)
                }
        
        statistics.update({
            'num_labels': len(labels),
            'label_type': 'binary' if labels.dim() == 1 else 'multi-class'
        })
        
        is_valid = len(issues) == 0
        
        return ValidationResult(is_valid, issues, warnings, statistics)
    
    def validate_multi_modal_consistency(self, 
                                       graph_data: Dict,
                                       temporal_data: torch.Tensor,
                                       behavioral_data: torch.Tensor,
                                       labels: torch.Tensor) -> ValidationResult:
        """Validate consistency across different data modalities"""
        issues = []
        warnings = []
        statistics = {}
        
        # Check sample size consistency
        num_graph_samples = graph_data['x'].size(0) if 'x' in graph_data else 0
        num_temporal_samples = temporal_data.size(0) if temporal_data is not None else 0
        num_behavioral_samples = behavioral_data.size(0) if behavioral_data is not None else 0
        num_labels = len(labels) if labels is not None else 0
        
        sample_counts = [num_graph_samples, num_temporal_samples, num_behavioral_samples, num_labels]
        sample_counts = [c for c in sample_counts if c > 0]  # Remove zeros (optional modalities)
        
        if len(set(sample_counts)) > 1:
            issues.append(f"Inconsistent sample sizes: graph={num_graph_samples}, "
                         f"temporal={num_temporal_samples}, behavioral={num_behavioral_samples}, "
                         f"labels={num_labels}")
        
        # Check temporal sequence alignment
        if temporal_data is not None and graph_data is not None:
            if num_temporal_samples != num_graph_samples:
                warnings.append("Temporal and graph data have different sample sizes")
        
        statistics.update({
            'graph_samples': num_graph_samples,
            'temporal_samples': num_temporal_samples,
            'behavioral_samples': num_behavioral_samples,
            'label_samples': num_labels
        })
        
        is_valid = len(issues) == 0
        
        return ValidationResult(is_valid, issues, warnings, statistics)
    
    def _calculate_graph_density(self, num_nodes: int, num_edges: int) -> float:
        """Calculate graph density"""
        if num_nodes <= 1:
            return 0.0
        max_edges = num_nodes * (num_nodes - 1)
        return num_edges / max_edges if max_edges > 0 else 0.0
    
    def generate_validation_report(self, validation_results: Dict[str, ValidationResult]) -> str:
        """Generate comprehensive validation report"""
        report = ["CyberShieldNet Data Validation Report", "=" * 50]
        
        for data_type, result in validation_results.items():
            report.append(f"\n{data_type.upper()} VALIDATION:")
            report.append(f"  Valid: {result.is_valid}")
            report.append(f"  Issues: {len(result.issues)}")
            report.append(f"  Warnings: {len(result.warnings)}")
            
            if result.issues:
                report.append("  Detailed Issues:")
                for issue in result.issues:
                    report.append(f"    - {issue}")
            
            if result.warnings:
                report.append("  Warnings:")
                for warning in result.warnings:
                    report.append(f"    - {warning}")
            
            if result.statistics:
                report.append("  Statistics:")
                for key, value in result.statistics.items():
                    report.append(f"    {key}: {value}")
        
        # Overall assessment
        all_valid = all(result.is_valid for result in validation_results.values())
        total_issues = sum(len(result.issues) for result in validation_results.values())
        total_warnings = sum(len(result.warnings) for result in validation_results.values())
        
        report.append(f"\nOVERALL ASSESSMENT:")
        report.append(f"  All Data Valid: {all_valid}")
        report.append(f"  Total Issues: {total_issues}")
        report.append(f"  Total Warnings: {total_warnings}")
        report.append(f"  Data Quality: {'PASS' if all_valid else 'FAIL'}")
        
        return "\n".join(report)