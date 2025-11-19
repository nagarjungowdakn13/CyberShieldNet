from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

try:
    import torch
except Exception:  # pragma: no cover - runtime import guard
    torch = None

TORCH_AVAILABLE = torch is not None

logger = logging.getLogger(__name__)

class ThreatDetectionMetrics:
    """
    Comprehensive metrics for cyber threat detection evaluation
    """
    
    def __init__(self, config: Dict):
        if not TORCH_AVAILABLE:
            raise RuntimeError("ThreatDetectionMetrics requires PyTorch. Install torch to compute metrics.")
        self.config = config
        self.metrics_history = {}
        
    def compute_binary_metrics(self, 
                             y_true: torch.Tensor, 
                             y_pred: torch.Tensor,
                             y_prob: Optional[torch.Tensor] = None) -> Dict:
        """Compute binary classification metrics for threat detection"""
        logger.info("Computing binary classification metrics...")
        
        y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
        metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='binary', zero_division=0
        )
        
        # Probability-based metrics
        if y_prob is not None:
            y_prob_np = y_prob.cpu().numpy() if torch.is_tensor(y_prob) else y_prob
            try:
                metrics['auc_roc'] = roc_auc_score(y_true_np, y_prob_np)
            except ValueError:
                metrics['auc_roc'] = 0.0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Threat-specific metrics
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        metrics['threat_detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return metrics
    
    def compute_multi_class_metrics(self, 
                                  y_true: torch.Tensor, 
                                  y_pred: torch.Tensor,
                                  class_names: List[str]) -> Dict:
        """Compute multi-class classification metrics"""
        logger.info("Computing multi-class classification metrics...")
        
        y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_np, y_pred_np, average=None, zero_division=0
        )
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(class_names):
            metrics['per_class'][class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        # Macro and weighted averages
        metrics['macro_precision'] = precision.mean()
        metrics['macro_recall'] = recall.mean()
        metrics['macro_f1'] = f1.mean()
        
        metrics['weighted_precision'] = np.average(precision, weights=support)
        metrics['weighted_recall'] = np.average(recall, weights=support)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        
        return metrics
    
    def compute_risk_assessment_metrics(self, 
                                      predicted_risk: torch.Tensor,
                                      actual_impact: torch.Tensor,
                                      risk_thresholds: Dict) -> Dict:
        """Compute metrics for risk assessment performance"""
        logger.info("Computing risk assessment metrics...")
        
        metrics = {}
        
        # Correlation between predicted risk and actual impact
        correlation = torch.corrcoef(torch.stack([predicted_risk, actual_impact]))[0, 1]
        metrics['risk_impact_correlation'] = correlation.item() if not torch.isnan(correlation) else 0.0
        
        # Risk calibration metrics
        metrics['calibration_error'] = self._compute_calibration_error(predicted_risk, actual_impact)
        
        # Risk level accuracy
        predicted_levels = self._classify_risk_levels(predicted_risk, risk_thresholds)
        actual_levels = self._classify_risk_levels(actual_impact, risk_thresholds)
        
        risk_accuracy = (predicted_levels == actual_levels).float().mean()
        metrics['risk_level_accuracy'] = risk_accuracy.item()
        
        return metrics
    
    def compute_temporal_metrics(self, 
                               predictions: torch.Tensor,
                               targets: torch.Tensor,
                               timestamps: List) -> Dict:
        """Compute temporal performance metrics"""
        logger.info("Computing temporal metrics...")
        
        metrics = {}
        
        # Early detection metrics
        early_detection_times = self._compute_early_detection_times(predictions, targets, timestamps)
        metrics['mean_early_detection_time'] = np.mean(early_detection_times) if early_detection_times else 0.0
        
        # Temporal stability
        temporal_stability = self._compute_temporal_stability(predictions)
        metrics['temporal_stability'] = temporal_stability
        
        # Prediction latency
        prediction_latency = self._compute_prediction_latency(predictions, timestamps)
        metrics['mean_prediction_latency'] = prediction_latency
        
        return metrics
    
    def _compute_calibration_error(self, predicted_risk: torch.Tensor, actual_impact: torch.Tensor) -> float:
        """Compute risk calibration error"""
        # Binned calibration error
        bins = torch.linspace(0, 1, 11)
        bin_indices = torch.bucketize(predicted_risk, bins)
        
        calibration_error = 0.0
        for i in range(len(bins) - 1):
            mask = bin_indices == i + 1
            if mask.sum() > 0:
                bin_mean_pred = predicted_risk[mask].mean()
                bin_mean_actual = actual_impact[mask].mean()
                calibration_error += torch.abs(bin_mean_pred - bin_mean_actual)
        
        return calibration_error.item() / (len(bins) - 1)
    
    def _classify_risk_levels(self, risk_scores: torch.Tensor, thresholds: Dict) -> torch.Tensor:
        """Classify risk scores into levels"""
        risk_levels = torch.zeros_like(risk_scores, dtype=torch.long)
        
        for i, (level, (min_val, max_val)) in enumerate(thresholds.items()):
            mask = (risk_scores >= min_val) & (risk_scores <= max_val)
            risk_levels[mask] = i
        
        return risk_levels
    
    def _compute_early_detection_times(self, 
                                     predictions: torch.Tensor,
                                     targets: torch.Tensor,
                                     timestamps: List) -> List[float]:
        """Compute early detection times for threats"""
        early_times = []
        
        threat_indices = torch.where(targets > 0.5)[0]
        
        for idx in threat_indices:
            # Find when threat was first predicted
            threat_time = timestamps[idx]
            for i in range(max(0, idx - 10), idx):  # Look back 10 time steps
                if predictions[i] > 0.5:
                    early_time = (threat_time - timestamps[i]).total_seconds()
                    early_times.append(early_time)
                    break
        
        return early_times
    
    def _compute_temporal_stability(self, predictions: torch.Tensor) -> float:
        """Compute temporal stability of predictions"""
        if len(predictions) < 2:
            return 1.0
        
        changes = torch.diff(predictions)
        stability = 1.0 - torch.mean(torch.abs(changes))
        return stability.item()
    
    def _compute_prediction_latency(self, predictions: torch.Tensor, timestamps: List) -> float:
        """Compute average prediction latency"""
        if len(timestamps) < 2:
            return 0.0
        
        latencies = []
        for i in range(1, len(timestamps)):
            latency = (timestamps[i] - timestamps[i-1]).total_seconds()
            latencies.append(latency)
        
        return np.mean(latencies) if latencies else 0.0

class AdversarialRobustnessMetrics:
    """Metrics for evaluating adversarial robustness"""
    
    def __init__(self):
        self.attack_success_rates = {}
        
    def compute_robustness_metrics(self, 
                                 clean_accuracy: float,
                                 adversarial_accuracy: float,
                                 attack_type: str) -> Dict:
        """Compute adversarial robustness metrics"""
        metrics = {}
        
        metrics['clean_accuracy'] = clean_accuracy
        metrics['adversarial_accuracy'] = adversarial_accuracy
        metrics['robustness_gap'] = clean_accuracy - adversarial_accuracy
        metrics['robustness_ratio'] = adversarial_accuracy / clean_accuracy if clean_accuracy > 0 else 0.0
        
        # Attack success rate
        attack_success_rate = 1.0 - (adversarial_accuracy / clean_accuracy) if clean_accuracy > 0 else 1.0
        metrics['attack_success_rate'] = attack_success_rate
        self.attack_success_rates[attack_type] = attack_success_rate
        
        return metrics
    
    def compute_transfer_attack_metrics(self, 
                                      original_model_accuracy: float,
                                      transferred_accuracy: float) -> Dict:
        """Compute transfer attack metrics"""
        metrics = {}
        
        metrics['transfer_success_rate'] = 1.0 - (transferred_accuracy / original_model_accuracy)
        metrics['transfer_effectiveness'] = transferred_accuracy / original_model_accuracy
        
        return metrics

class EnergyEfficiencyMetrics:
    """Metrics for computational and energy efficiency"""
    
    def __init__(self):
        self.energy_consumption = 0.0
        self.inference_times = []
        
    def record_inference_time(self, inference_time: float):
        """Record inference time for a batch"""
        self.inference_times.append(inference_time)
    
    def record_energy_consumption(self, energy_used: float):
        """Record energy consumption"""
        self.energy_consumption += energy_used
    
    def compute_efficiency_metrics(self, 
                                 model_size: float,
                                 throughput: float) -> Dict:
        """Compute energy efficiency metrics"""
        metrics = {}
        
        # Computational efficiency
        metrics['model_size_mb'] = model_size
        metrics['throughput_eps'] = throughput  # events per second
        
        if self.inference_times:
            metrics['mean_inference_time'] = np.mean(self.inference_times)
            metrics['inference_time_std'] = np.std(self.inference_times)
        
        # Energy efficiency
        metrics['total_energy_consumption'] = self.energy_consumption
        metrics['energy_per_prediction'] = self.energy_consumption / len(self.inference_times) if self.inference_times else 0.0
        
        # Carbon footprint (approximate)
        metrics['carbon_emission_kg'] = self.energy_consumption * 0.5  # kg CO2 per kWh (approximate)
        
        return metrics

def generate_comprehensive_report(metrics: Dict, title: str = "CyberShieldNet Performance Report") -> str:
    """Generate comprehensive performance report"""
    report = [title, "=" * len(title), ""]
    
    for category, category_metrics in metrics.items():
        report.append(f"{category.upper()} METRICS:")
        report.append("-" * len(category.upper() + " METRICS:"))
        
        for metric_name, metric_value in category_metrics.items():
            if isinstance(metric_value, dict):
                report.append(f"  {metric_name}:")
                for sub_name, sub_value in metric_value.items():
                    report.append(f"    {sub_name}: {sub_value:.4f}")
            else:
                report.append(f"  {metric_name}: {metric_value:.4f}")
        
        report.append("")
    
    return "\n".join(report)