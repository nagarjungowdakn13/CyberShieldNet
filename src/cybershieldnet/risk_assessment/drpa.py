import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)

class DynamicRiskPropagation(nn.Module):
    """
    Dynamic Risk Propagation Algorithm (DRPA)
    Quantifies and propagates organizational risk in real-time
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Risk parameters
        self.risk_decay_factor = config.get('risk_decay_factor', 0.85)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.propagation_depth = config.get('propagation_depth', 5)
        self.context_weight = config.get('context_weight', 0.15)
        
        # Risk component weights
        self.weights = nn.ParameterDict({
            'threat_probability': nn.Parameter(torch.tensor(config.get('weights', {}).get('threat_probability', 0.4))),
            'impact': nn.Parameter(torch.tensor(config.get('weights', {}).get('impact', 0.3))),
            'vulnerability': nn.Parameter(torch.tensor(config.get('weights', {}).get('vulnerability', 0.2))),
            'context': nn.Parameter(torch.tensor(config.get('weights', {}).get('context', 0.1)))
        })
        
        # Risk propagation network
        self.propagation_network = RiskPropagationNetwork(
            input_dim=4,  # threat, impact, vulnerability, context
            hidden_dims=[64, 32, 16],
            propagation_depth=self.propagation_depth
        )
        
        self.asset_similarity = AssetSimilarityNetwork()
        self.context_encoder = ContextEncoder()
        
        logger.info("DynamicRiskPropagation initialized")
    
    def forward(self, 
                threat_predictions: torch.Tensor,
                assets: Dict,
                vulnerabilities: Dict) -> torch.Tensor:
        """
        Compute dynamic risk scores for organizational assets
        
        Args:
            threat_predictions: Threat probability scores (batch_size, num_assets)
            assets: Asset information dictionary
            vulnerabilities: Vulnerability data dictionary
            
        Returns:
            Risk scores for all assets (batch_size, num_assets)
        """
        batch_size, num_assets = threat_predictions.shape
        
        # Compute base risk components
        threat_risk = self._compute_threat_risk(threat_predictions)
        impact_risk = self._compute_impact_risk(assets)
        vulnerability_risk = self._compute_vulnerability_risk(vulnerabilities)
        context_risk = self._compute_context_risk(assets)
        
        # Combine risk components
        base_risk = self._combine_risk_components(
            threat_risk, impact_risk, vulnerability_risk, context_risk
        )
        
        # Risk propagation through organizational network
        propagated_risk = self.propagation_network(
            base_risk, assets, self.risk_decay_factor
        )
        
        return propagated_risk
    
    def _compute_threat_risk(self, threat_predictions: torch.Tensor) -> torch.Tensor:
        """Compute risk from threat predictions"""
        # Apply non-linear transformation to emphasize high probabilities
        threat_risk = torch.sigmoid(threat_predictions * 3)  # Sharpens probabilities
        return threat_risk
    
    def _compute_impact_risk(self, assets: Dict) -> torch.Tensor:
        """Compute risk from business impact assessment"""
        impact_scores = assets.get('criticality', torch.ones_like(list(assets.values())[0]))
        
        # Normalize impact scores to [0, 1]
        if impact_scores.max() > 0:
            impact_risk = impact_scores / impact_scores.max()
        else:
            impact_risk = impact_scores
        
        return impact_risk
    
    def _compute_vulnerability_risk(self, vulnerabilities: Dict) -> torch.Tensor:
        """Compute risk from vulnerability data"""
        if 'severity' in vulnerabilities:
            vuln_scores = vulnerabilities['severity']
            # Normalize to [0, 1]
            vuln_risk = vuln_scores / 10.0  # Assuming CVSS-like scores
        else:
            vuln_risk = torch.zeros_like(list(vulnerabilities.values())[0])
        
        return vuln_risk
    
    def _compute_context_risk(self, assets: Dict) -> torch.Tensor:
        """Compute risk from contextual factors"""
        context_features = self.context_encoder(assets)
        context_risk = torch.sigmoid(context_features)
        return context_risk
    
    def _combine_risk_components(self,
                               threat_risk: torch.Tensor,
                               impact_risk: torch.Tensor,
                               vulnerability_risk: torch.Tensor,
                               context_risk: torch.Tensor) -> torch.Tensor:
        """Combine different risk components with learned weights"""
        # Ensure weights sum to 1
        total_weight = sum([weight for weight in self.weights.values()])
        normalized_weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Weighted combination
        combined_risk = (
            normalized_weights['threat_probability'] * threat_risk +
            normalized_weights['impact'] * impact_risk +
            normalized_weights['vulnerability'] * vulnerability_risk +
            normalized_weights['context'] * context_risk
        )
        
        return combined_risk
    
    def propagate_risk(self, 
                      risk_scores: torch.Tensor,
                      asset_connections: torch.Tensor,
                      similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Propagate risk through organizational network
        
        Args:
            risk_scores: Initial risk scores
            asset_connections: Adjacency matrix of asset connections
            similarity_matrix: Similarity matrix between assets
            
        Returns:
            Propagated risk scores
        """
        current_risk = risk_scores.clone()
        
        for step in range(self.propagation_depth):
            # Compute risk propagation
            propagated = torch.matmul(asset_connections, current_risk)
            
            # Apply decay and similarity
            decayed_risk = propagated * (self.risk_decay_factor ** (step + 1))
            similarity_adjusted = decayed_risk * similarity_matrix
            
            # Update current risk (maximum of current and propagated)
            current_risk = torch.max(current_risk, similarity_adjusted)
        
        return current_risk

class RiskPropagationNetwork(nn.Module):
    """Neural network for risk propagation"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], propagation_depth: int):
        super().__init__()
        
        self.propagation_depth = propagation_depth
        
        # Propagation layers
        self.propagation_layers = nn.ModuleList()
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.propagation_layers.append(
                nn.Linear(current_dim, hidden_dim)
            )
            current_dim = hidden_dim
        
        self.output_layer = nn.Linear(current_dim, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, base_risk: torch.Tensor, assets: Dict, decay_factor: float) -> torch.Tensor:
        """Propagate risk through network"""
        # Extract features for propagation
        features = self._extract_propagation_features(base_risk, assets)
        
        # Apply propagation layers
        x = features
        for layer in self.propagation_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        
        # Final risk score
        risk_scores = torch.sigmoid(self.output_layer(x)).squeeze(-1)
        
        return risk_scores
    
    def _extract_propagation_features(self, base_risk: torch.Tensor, assets: Dict) -> torch.Tensor:
        """Extract features for risk propagation"""
        features = [base_risk.unsqueeze(-1)]
        
        # Add asset-specific features
        if 'criticality' in assets:
            features.append(assets['criticality'].unsqueeze(-1))
        if 'connectivity' in assets:
            features.append(assets['connectivity'].unsqueeze(-1))
        if 'sensitivity' in assets:
            features.append(assets['sensitivity'].unsqueeze(-1))
        
        # Combine features
        combined = torch.cat(features, dim=-1)
        return combined

class AssetSimilarityNetwork(nn.Module):
    """Compute similarity between organizational assets"""
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        self.feature_projection = nn.Linear(feature_dim, 32)
        self.similarity_net = nn.Sequential(
            nn.Linear(64, 32),  # Concatenated features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, asset_features: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix between assets"""
        num_assets = asset_features.size(0)
        
        # Project features
        projected = self.feature_projection(asset_features)
        
        # Compute pairwise similarities
        similarity_matrix = torch.zeros(num_assets, num_assets, device=asset_features.device)
        
        for i in range(num_assets):
            for j in range(num_assets):
                if i != j:
                    # Concatenate features and compute similarity
                    pair_features = torch.cat([projected[i], projected[j]])
                    similarity = self.similarity_net(pair_features.unsqueeze(0))
                    similarity_matrix[i, j] = similarity.squeeze()
        
        # Set self-similarity to 1
        similarity_matrix.fill_diagonal_(1.0)
        
        return similarity_matrix

class ContextEncoder(nn.Module):
    """Encode contextual information for risk assessment"""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 8):
        super().__init__()
        
        self.context_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, assets: Dict) -> torch.Tensor:
        """Encode contextual factors"""
        context_features = []
        
        # Extract relevant context features
        if 'business_value' in assets:
            context_features.append(assets['business_value'].unsqueeze(-1))
        if 'data_sensitivity' in assets:
            context_features.append(assets['data_sensitivity'].unsqueeze(-1))
        if 'compliance_requirements' in assets:
            context_features.append(assets['compliance_requirements'].unsqueeze(-1))
        if 'recovery_time' in assets:
            # Invert recovery time (shorter recovery = higher risk)
            recovery_risk = 1.0 / (assets['recovery_time'] + 1e-6)
            context_features.append(recovery_risk.unsqueeze(-1))
        
        # Default context if no features available
        if not context_features:
            default_context = torch.ones_like(list(assets.values())[0]).unsqueeze(-1)
            context_features.append(default_context)
        
        # Combine context features
        combined = torch.cat(context_features, dim=-1)
        
        # Encode context
        context_score = self.context_net(combined)
        
        return context_score.squeeze(-1)

class RiskVisualizer:
    """Visualize and analyze risk propagation"""
    
    def __init__(self):
        self.risk_levels = {
            'CRITICAL': (0.9, 1.0),
            'HIGH': (0.7, 0.9),
            'MEDIUM': (0.4, 0.7),
            'LOW': (0.1, 0.4),
            'NONE': (0.0, 0.1)
        }
    
    def classify_risk_levels(self, risk_scores: torch.Tensor) -> List[str]:
        """Classify risk scores into levels"""
        risk_levels = []
        
        for score in risk_scores:
            score_val = score.item()
            for level, (min_val, max_val) in self.risk_levels.items():
                if min_val <= score_val <= max_val:
                    risk_levels.append(level)
                    break
            else:
                risk_levels.append('UNKNOWN')
        
        return risk_levels
    
    def generate_risk_report(self, 
                           risk_scores: torch.Tensor,
                           assets: List[str],
                           propagation_paths: Dict) -> str:
        """Generate comprehensive risk assessment report"""
        report = ["CyberShieldNet Risk Assessment Report", "=" * 50]
        
        # Overall risk summary
        overall_risk = risk_scores.mean().item()
        report.append(f"Overall Organizational Risk: {overall_risk:.3f}")
        report.append(f"Highest Risk Asset: {assets[risk_scores.argmax().item()]} "
                     f"(Score: {risk_scores.max().item():.3f})")
        
        # Risk level distribution
        risk_levels = self.classify_risk_levels(risk_scores)
        level_counts = {}
        for level in risk_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        report.append("\nRisk Level Distribution:")
        for level, count in level_counts.items():
            percentage = (count / len(risk_scores)) * 100
            report.append(f"  {level}: {count} assets ({percentage:.1f}%)")
        
        # Top risky assets
        report.append("\nTop 10 Riskiest Assets:")
        sorted_indices = torch.argsort(risk_scores, descending=True)
        for i, idx in enumerate(sorted_indices[:10]):
            asset_name = assets[idx.item()]
            risk_score = risk_scores[idx].item()
            risk_level = risk_levels[idx.item()]
            report.append(f"  {i+1}. {asset_name}: {risk_score:.3f} ({risk_level})")
        
        # Risk propagation analysis
        report.append("\nRisk Propagation Analysis:")
        for asset, paths in list(propagation_paths.items())[:5]:  # Show top 5
            report.append(f"  {asset}: {len(paths)} propagation paths")
        
        return "\n".join(report)