import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RiskPropagation:
    """
    Risk propagation algorithms for organizational networks
    """

    def __init__(self, config: Dict):
        self.config = config
        self.decay_factor = config.get('risk_decay_factor', 0.85)
        self.max_propagation_steps = config.get('propagation_depth', 5)

    def propagate_risk(self, 
                      initial_risk: torch.Tensor,
                      graph: nx.Graph,
                      asset_features: Dict) -> torch.Tensor:
        """
        Propagate risk through the organizational network

        Args:
            initial_risk: Initial risk scores for each node
            graph: NetworkX graph representing organizational structure
            asset_features: Additional features for assets

        Returns:
            Propagated risk scores
        """
        num_nodes = len(graph.nodes())
        risk_scores = initial_risk.clone()

        # Create adjacency matrix
        adj_matrix = self._create_adjacency_matrix(graph, num_nodes)

        # Propagate risk for multiple steps
        for step in range(self.max_propagation_steps):
            new_risk = torch.zeros_like(risk_scores)

            for i in range(num_nodes):
                # Risk from neighbors
                neighbor_risk = 0.0
                for j in range(num_nodes):
                    if adj_matrix[i, j] > 0:
                        # Decayed risk from neighbor j to i
                        decayed_risk = risk_scores[j] * self.decay_factor
                        # Adjust by similarity and connectivity
                        similarity = self._compute_asset_similarity(
                            asset_features, i, j
                        )
                        neighbor_risk += decayed_risk * similarity

                # Update risk: maximum of current risk and propagated risk
                new_risk[i] = max(risk_scores[i], neighbor_risk)

            risk_scores = new_risk

        return risk_scores

    def _create_adjacency_matrix(self, graph: nx.Graph, num_nodes: int) -> torch.Tensor:
        """Create adjacency matrix from NetworkX graph"""
        adj_matrix = torch.zeros((num_nodes, num_nodes))

        for i, j in graph.edges():
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Undirected graph

        return adj_matrix

    def _compute_asset_similarity(self, asset_features: Dict, i: int, j: int) -> float:
        """Compute similarity between two assets"""
        # Example similarity based on asset features
        similarity = 1.0  # Default similarity

        if 'asset_type' in asset_features:
            if asset_features['asset_type'][i] == asset_features['asset_type'][j]:
                similarity *= 1.2  # Increase similarity for same type

        if 'criticality' in asset_features:
            # Similar criticality leads to higher similarity
            crit_i = asset_features['criticality'][i]
            crit_j = asset_features['criticality'][j]
            similarity *= 1.0 - abs(crit_i - crit_j)  # Decrease by difference

        return max(0.1, similarity)  # Ensure minimum similarity

class MarkovRiskPropagation:
    """
    Markov-based risk propagation model
    """

    def __init__(self, config: Dict):
        self.config = config
        self.transition_matrix = None

    def build_transition_matrix(self, graph: nx.Graph, asset_features: Dict):
        """Build Markov transition matrix for risk propagation"""
        num_nodes = len(graph.nodes())
        transition_matrix = torch.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            neighbors = list(graph.neighbors(i))
            if not neighbors:
                continue

            # Transition probability based on similarity and connectivity
            total_similarity = 0.0
            similarities = []

            for j in neighbors:
                similarity = self._compute_similarity(asset_features, i, j)
                similarities.append(similarity)
                total_similarity += similarity

            # Normalize to create probability distribution
            for idx, j in enumerate(neighbors):
                transition_matrix[i, j] = similarities[idx] / total_similarity

        self.transition_matrix = transition_matrix

    def propagate_risk(self, initial_risk: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Propagate risk using Markov process"""
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not built. Call build_transition_matrix first.")

        current_risk = initial_risk.clone()

        for step in range(steps):
            current_risk = torch.matmul(self.transition_matrix, current_risk)

        return current_risk

    def _compute_similarity(self, asset_features: Dict, i: int, j: int) -> float:
        """Compute similarity between two assets for Markov transitions"""
        similarity = 1.0

        # Example factors for similarity
        if 'department' in asset_features:
            if asset_features['department'][i] == asset_features['department'][j]:
                similarity *= 2.0

        if 'security_level' in asset_features:
            level_i = asset_features['security_level'][i]
            level_j = asset_features['security_level'][j]
            # Closer security levels have higher similarity
            similarity *= 1.0 / (1.0 + abs(level_i - level_j))

        return similarity

class BayesianRiskNetwork:
    """
    Bayesian network for risk assessment
    """

    def __init__(self, config: Dict):
        self.config = config
        self.risk_factors = config.get('risk_factors', [])
        self.conditional_probs = {}

    def learn_conditional_probabilities(self, data: torch.Tensor, labels: torch.Tensor):
        """Learn conditional probabilities from data"""
        # This is a simplified implementation
        # In practice, you would use a more sophisticated Bayesian learning approach

        for factor in self.risk_factors:
            # Example: Learn P(risk | factor)
            factor_data = data[:, factor]
            risk_given_factor = self._compute_conditional_prob(factor_data, labels)
            self.conditional_probs[factor] = risk_given_factor

    def infer_risk(self, evidence: Dict) -> float:
        """Infer risk given evidence using Bayesian network"""
        risk_prob = 1.0

        for factor, value in evidence.items():
            if factor in self.conditional_probs:
                # Look up conditional probability
                prob = self.conditional_probs[factor].get(value, 0.5)
                risk_prob *= prob

        return risk_prob

    def _compute_conditional_prob(self, factor_data: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Compute conditional probabilities for a factor"""
        unique_values = torch.unique(factor_data)
        conditional_probs = {}

        for value in unique_values:
            value_mask = factor_data == value
            risk_given_value = labels[value_mask].mean()
            conditional_probs[value.item()] = risk_given_value.item()

        return conditional_probs