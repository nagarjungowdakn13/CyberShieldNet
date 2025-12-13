import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RiskCalculator:
    """
    Comprehensive risk calculator for cyber threats
    """

    def __init__(self, config: Dict):
        self.config = config
        self.risk_components = config.get('risk_components', ['threat', 'vulnerability', 'impact'])

    def calculate_aggregate_risk(self, component_risks: Dict) -> torch.Tensor:
        """Calculate aggregate risk from components"""
        aggregate_risk = torch.zeros_like(list(component_risks.values())[0])

        for component in self.risk_components:
            if component in component_risks:
                aggregate_risk += component_risks[component]

        # Normalize to [0, 1]
        aggregate_risk = torch.clamp(aggregate_risk, 0, 1)

        return aggregate_risk

    def calculate_threat_risk(self, threat_data: Dict) -> torch.Tensor:
        """Calculate risk from threat intelligence"""
        threat_indicators = threat_data.get('indicators', [])
        threat_confidence = threat_data.get('confidence', 1.0)

        threat_risk = torch.zeros(len(threat_indicators))

        for i, indicator in enumerate(threat_indicators):
            # Example threat risk calculation
            indicator_risk = self._evaluate_threat_indicator(indicator)
            threat_risk[i] = indicator_risk * threat_confidence

        return threat_risk

    def calculate_vulnerability_risk(self, vulnerability_data: Dict) -> torch.Tensor:
        """Calculate risk from vulnerability data"""
        vuln_scores = vulnerability_data.get('scores', [])
        exploitability = vulnerability_data.get('exploitability', 1.0)

        vuln_risk = torch.tensor(vuln_scores) * exploitability

        return vuln_risk

    def calculate_impact_risk(self, impact_data: Dict) -> torch.Tensor:
        """Calculate risk from business impact assessment"""
        financial_impact = impact_data.get('financial', 0.0)
        operational_impact = impact_data.get('operational', 0.0)
        reputational_impact = impact_data.get('reputational', 0.0)

        # Normalize impacts to [0, 1] range
        max_financial = impact_data.get('max_financial', 1e6)
        financial_risk = min(financial_impact / max_financial, 1.0)

        operational_risk = min(operational_impact / 10.0, 1.0)  # Assuming scale 1-10
        reputational_risk = min(reputational_impact / 10.0, 1.0)

        # Combined impact risk
        impact_risk = (financial_risk + operational_risk + reputational_risk) / 3.0

        return torch.tensor(impact_risk)

    def _evaluate_threat_indicator(self, indicator: Dict) -> float:
        """Evaluate a single threat indicator"""
        indicator_type = indicator.get('type', 'unknown')
        severity = indicator.get('severity', 0.0)
        relevance = indicator.get('relevance', 1.0)

        base_risk = severity * relevance

        # Adjust based on indicator type
        type_weights = {
            'malware': 1.2,
            'phishing': 1.1,
            'ddos': 1.3,
            'insider': 1.4,
            'unknown': 1.0
        }

        weight = type_weights.get(indicator_type, 1.0)
        return base_risk * weight

class QuantitativeRiskAnalyzer:
    """
    Quantitative risk analysis using FAIR methodology
    """

    def __init__(self, config: Dict):
        self.config = config

    def calculate_loss_metrics(self, threat_data: Dict, vulnerability_data: Dict) -> Dict:
        """Calculate loss metrics using quantitative analysis"""
        # Frequency of threat events
        threat_frequency = self._estimate_threat_frequency(threat_data)

        # Vulnerability and control effectiveness
        vulnerability = self._assess_vulnerability(vulnerability_data)

        # Impact estimation
        impact = self._estimate_impact(threat_data)

        # Calculate loss metrics
        probable_loss = threat_frequency * vulnerability * impact

        return {
            'probable_loss': probable_loss,
            'threat_frequency': threat_frequency,
            'vulnerability': vulnerability,
            'impact': impact
        }

    def _estimate_threat_frequency(self, threat_data: Dict) -> float:
        """Estimate frequency of threat events"""
        historical_events = threat_data.get('historical_events', [])
        threat_intel = threat_data.get('threat_intel', {})

        # Simple frequency estimation
        if historical_events:
            frequency = len(historical_events) / 365  # Events per day
        else:
            # Use threat intelligence to estimate
            frequency = threat_intel.get('estimated_frequency', 0.01)

        return frequency

    def _assess_vulnerability(self, vulnerability_data: Dict) -> float:
        """Assess vulnerability level"""
        vuln_scores = vulnerability_data.get('scores', [])
        controls = vulnerability_data.get('controls', [])

        if vuln_scores:
            avg_vulnerability = sum(vuln_scores) / len(vuln_scores)
        else:
            avg_vulnerability = 0.5

        # Adjust for control effectiveness
        control_effectiveness = sum(controls) / len(controls) if controls else 0.0
        adjusted_vulnerability = avg_vulnerability * (1 - control_effectiveness)

        return adjusted_vulnerability

    def _estimate_impact(self, threat_data: Dict) -> float:
        """Estimate impact of threat realization"""
        financial_impact = threat_data.get('financial_impact', 0.0)
        operational_impact = threat_data.get('operational_impact', 0.0)

        # Normalize impacts
        max_financial = threat_data.get('max_financial_impact', 1e6)
        normalized_financial = financial_impact / max_financial

        normalized_operational = operational_impact / 10.0  # Scale 1-10

        # Combined impact
        total_impact = (normalized_financial + normalized_operational) / 2.0

        return total_impact

class RiskPrioritization:
    """
    Risk prioritization and ranking
    """

    def __init__(self, config: Dict):
        self.config = config
        self.prioritization_method = config.get('prioritization_method', 'weighted_score')

    def prioritize_risks(self, risks: torch.Tensor, assets: List[str]) -> List[Tuple[str, float]]:
        """Prioritize risks based on scores and business context"""
        if self.prioritization_method == 'weighted_score':
            prioritized = self._weighted_prioritization(risks, assets)
        elif self.prioritization_method == 'criticality_based':
            prioritized = self._criticality_prioritization(risks, assets)
        else:
            prioritized = self._default_prioritization(risks, assets)

        return prioritized

    def _weighted_prioritization(self, risks: torch.Tensor, assets: List[str]) -> List[Tuple[str, float]]:
        """Weighted prioritization considering multiple factors"""
        prioritized = []

        for i, asset in enumerate(assets):
            risk_score = risks[i].item()
            # Additional factors could be considered here
            prioritized.append((asset, risk_score))

        # Sort by risk score descending
        prioritized.sort(key=lambda x: x[1], reverse=True)

        return prioritized

    def _criticality_prioritization(self, risks: torch.Tensor, assets: List[str]) -> List[Tuple[str, float]]:
        """Prioritization based on asset criticality"""
        # This would require criticality information for assets
        # For now, use risk scores with a dummy criticality factor
        criticality_factors = torch.ones_like(risks)  # Placeholder

        adjusted_risks = risks * criticality_factors

        prioritized = []
        for i, asset in enumerate(assets):
            prioritized.append((asset, adjusted_risks[i].item()))

        prioritized.sort(key=lambda x: x[1], reverse=True)

        return prioritized

    def _default_prioritization(self, risks: torch.Tensor, assets: List[str]) -> List[Tuple[str, float]]:
        """Default prioritization by risk score"""
        prioritized = []

        for i, asset in enumerate(assets):
            prioritized.append((asset, risks[i].item()))

        prioritized.sort(key=lambda x: x[1], reverse=True)

        return prioritized