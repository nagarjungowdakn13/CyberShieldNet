import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """
    Explainable AI engine for threat intelligence predictions
    Provides SHAP analysis, attention visualization, and feature importance
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.shap_explainer = None
        self.feature_names = []
        
    def initialize_shap_explainer(self, model: nn.Module, background_data: torch.Tensor):
        """Initialize SHAP explainer with background data"""
        logger.info("Initializing SHAP explainer...")
        
        # Wrap model for SHAP compatibility
        def model_predict(x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return model(x).numpy()
        
        self.shap_explainer = shap.KernelExplainer(model_predict, background_data)
        logger.info("SHAP explainer initialized")
    
    def compute_shap_values(self, input_data: torch.Tensor) -> np.ndarray:
        """Compute SHAP values for input data"""
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap_explainer first.")
        
        logger.info("Computing SHAP values...")
        shap_values = self.shap_explainer.shap_values(input_data.numpy())
        return shap_values
    
    def analyze_feature_importance(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict:
        """Analyze feature importance from SHAP values"""
        logger.info("Analyzing feature importance...")
        
        # Mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort features by importance
        feature_importance = {}
        for i, feature_name in enumerate(feature_names):
            feature_importance[feature_name] = mean_abs_shap[i]
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def generate_attention_weights(self, model: nn.Module, input_data: torch.Tensor) -> torch.Tensor:
        """Generate attention weights from model for interpretability"""
        logger.info("Generating attention weights...")
        
        # Hook to capture attention weights
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_weights.append(module.attention_weights.detach())
        
        # Register hooks on attention layers
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower():
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass to capture attention
        with torch.no_grad():
            _ = model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return torch.stack(attention_weights) if attention_weights else torch.tensor([])
    
    def visualize_shap_summary(self, shap_values: np.ndarray, feature_names: List[str], save_path: str = None):
        """Create SHAP summary plot"""
        logger.info("Creating SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.close()
    
    def visualize_attention_heatmap(self, attention_weights: torch.Tensor, labels: List[str], save_path: str = None):
        """Create attention heatmap visualization"""
        if attention_weights.numel() == 0:
            logger.warning("No attention weights to visualize")
            return
        
        logger.info("Creating attention heatmap...")
        
        # Use the first attention head for visualization
        attention_matrix = attention_weights[0].cpu().numpy()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(attention_matrix, 
                   xticklabels=labels,
                   yticklabels=labels,
                   cmap='viridis',
                   center=0)
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Target Assets')
        plt.ylabel('Source Assets')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Attention heatmap saved to {save_path}")
        
        plt.close()
    
    def generate_explanation_report(self, 
                                  shap_values: np.ndarray,
                                  feature_importance: Dict,
                                  prediction: float,
                                  threshold: float = 0.5) -> str:
        """Generate natural language explanation report"""
        logger.info("Generating explanation report...")
        
        report = []
        report.append("CyberShieldNet Threat Explanation Report")
        report.append("=" * 50)
        report.append(f"Prediction: {prediction:.3f} ({'Threat' if prediction > threshold else 'Normal'})")
        report.append("")
        
        # Top contributing features
        report.append("Top Contributing Features:")
        top_features = list(feature_importance.items())[:5]
        for feature, importance in top_features:
            report.append(f"  - {feature}: {importance:.4f}")
        
        # Feature impact analysis
        report.append("")
        report.append("Feature Impact Analysis:")
        
        for i, (feature, importance) in enumerate(top_features):
            impact_level = "High" if importance > 0.1 else "Medium" if importance > 0.05 else "Low"
            report.append(f"  {i+1}. {feature}: {impact_level} impact ({importance:.3f})")
        
        # Risk factors
        report.append("")
        report.append("Key Risk Factors:")
        
        high_impact_features = [f for f, imp in feature_importance.items() if imp > 0.1]
        if high_impact_features:
            for feature in high_impact_features[:3]:
                report.append(f"  - {feature} significantly increases threat probability")
        else:
            report.append("  - No single feature dominates the prediction")
        
        return "\n".join(report)

class TemporalExplanation:
    """Temporal pattern explanation for sequence data"""
    
    def __init__(self):
        self.important_time_steps = []
    
    def analyze_temporal_importance(self, temporal_data: torch.Tensor, model: nn.Module) -> List[float]:
        """Analyze importance of different time steps"""
        logger.info("Analyzing temporal importance...")
        
        # Use gradient-based importance
        temporal_data.requires_grad_(True)
        
        # Forward pass
        output = model(temporal_data)
        
        # Backward pass to get gradients
        output.backward(torch.ones_like(output))
        
        # Compute importance as gradient magnitude
        temporal_importance = torch.norm(temporal_data.grad, dim=2).mean(dim=0)
        
        return temporal_importance.tolist()
    
    def visualize_temporal_patterns(self, 
                                  temporal_data: torch.Tensor,
                                  importance_weights: List[float],
                                  save_path: str = None):
        """Visualize temporal patterns with importance weights"""
        logger.info("Visualizing temporal patterns...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot temporal data
        time_steps = range(temporal_data.shape[1])
        mean_sequence = temporal_data.mean(dim=0).mean(dim=1)  # Average across batches and features
        
        ax1.plot(time_steps, mean_sequence, label='Average Sequence')
        ax1.set_title('Temporal Pattern')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Feature Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot importance weights
        ax2.bar(time_steps, importance_weights, alpha=0.7, color='red')
        ax2.set_title('Temporal Importance Weights')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Importance')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Temporal pattern visualization saved to {save_path}")
        
        plt.close()

class GraphExplanation:
    """Graph structure explanation for network data"""
    
    def __init__(self):
        self.node_importance = {}
        self.edge_importance = {}
    
    def analyze_graph_importance(self, graph_data: Dict, model: nn.Module) -> Dict:
        """Analyze importance of nodes and edges in graph"""
        logger.info("Analyzing graph importance...")
        
        # Node importance analysis
        node_features = graph_data['x']
        node_features.requires_grad_(True)
        
        # Forward pass
        output = model(graph_data)
        
        # Backward pass for node gradients
        output.backward(torch.ones_like(output))
        
        # Node importance as gradient magnitude
        node_importance = torch.norm(node_features.grad, dim=1)
        
        # Edge importance (simplified)
        edge_importance = torch.ones(graph_data['edge_index'].shape[1])
        
        return {
            'node_importance': node_importance.tolist(),
            'edge_importance': edge_importance.tolist()
        }
    
    def visualize_graph_importance(self, 
                                 graph_data: Dict,
                                 importance_scores: Dict,
                                 node_labels: List[str],
                                 save_path: str = None):
        """Visualize graph with importance coloring"""
        logger.info("Visualizing graph importance...")
        
        try:
            import networkx as nx
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes with importance
            for i, label in enumerate(node_labels):
                importance = importance_scores['node_importance'][i]
                G.add_node(i, importance=importance, label=label)
            
            # Add edges
            edge_index = graph_data['edge_index'].numpy()
            for i in range(edge_index.shape[1]):
                source, target = edge_index[0, i], edge_index[1, i]
                G.add_edge(source, target)
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Node colors based on importance
            node_colors = [G.nodes[n]['importance'] for n in G.nodes()]
            node_sizes = [300 + G.nodes[n]['importance'] * 1000 for n in G.nodes()]
            
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx_nodes(G, pos, 
                                 node_color=node_colors,
                                 node_size=node_sizes,
                                 cmap='Reds',
                                 alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.5)
            nx.draw_networkx_labels(G, pos, 
                                  {n: G.nodes[n]['label'] for n in G.nodes()},
                                  font_size=8)
            
            plt.title('Graph Importance Visualization')
            plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), label='Node Importance')
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Graph importance visualization saved to {save_path}")
            
            plt.close()
            
        except ImportError:
            logger.warning("NetworkX not available for graph visualization")