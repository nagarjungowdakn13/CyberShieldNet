import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class AdversarialDefense(nn.Module):
    """
    Adversarial defense mechanisms for robust threat detection
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        self.defense_mode = config.get('defense_mode', 'training')
        self.attack_types = config.get('attack_types', ['fgsm', 'pgd'])
        
        # Defense components
        self.input_sanitizer = InputSanitizer(config)
        self.feature_squeezer = FeatureSqueezer(config)
        self.gradient_masker = GradientMasker(config)

    def forward(self, x: torch.Tensor, model: nn.Module, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with adversarial defense"""
        if self.training and self.defense_mode == 'training':
            # Apply adversarial training
            x_adv = self._generate_adversarial_examples(x, model, targets)
            x = torch.cat([x, x_adv], dim=0)
            if targets is not None:
                targets = torch.cat([targets, targets], dim=0)

        # Input sanitization
        x_clean = self.input_sanitizer(x)
        
        # Feature squeezing
        x_squeezed = self.feature_squeezer(x_clean)
        
        return x_squeezed

    def _generate_adversarial_examples(self, 
                                     x: torch.Tensor, 
                                     model: nn.Module, 
                                     targets: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples for training"""
        adversarial_examples = []
        
        for attack_type in self.attack_types:
            if attack_type == 'fgsm':
                adv_x = self._fgsm_attack(x, model, targets)
            elif attack_type == 'pgd':
                adv_x = self._pgd_attack(x, model, targets)
            else:
                continue
                
            adversarial_examples.append(adv_x)
        
        return torch.cat(adversarial_examples, dim=0)

    def _fgsm_attack(self, x: torch.Tensor, model: nn.Module, targets: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        """Fast Gradient Sign Method attack"""
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        data_grad = x_adv.grad.data
        sign_data_grad = data_grad.sign()
        x_adv = x_adv + epsilon * sign_data_grad
        
        # Project back to valid range
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()

    def _pgd_attack(self, 
                   x: torch.Tensor, 
                   model: nn.Module, 
                   targets: torch.Tensor, 
                   epsilon: float = 0.1, 
                   alpha: float = 0.01, 
                   iterations: int = 10) -> torch.Tensor:
        """Projected Gradient Descent attack"""
        x_adv = x.clone().detach()
        
        for i in range(iterations):
            x_adv.requires_grad_(True)
            
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, targets)
            
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                # Update adversarial example
                grad = x_adv.grad.data
                x_adv = x_adv + alpha * grad.sign()
                
                # Project back to epsilon ball
                delta = torch.clamp(x_adv - x, -epsilon, epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()

class InputSanitizer(nn.Module):
    """Input sanitization for adversarial defense"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.sanitization_method = config.get('sanitization_method', 'anomaly_detection')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sanitize input to remove potential adversarial perturbations"""
        if self.sanitization_method == 'anomaly_detection':
            return self._anomaly_based_sanitization(x)
        elif self.sanitization_method == 'smoothing':
            return self._smoothing_sanitization(x)
        else:
            return x

    def _anomaly_based_sanitization(self, x: torch.Tensor) -> torch.Tensor:
        """Anomaly-based input sanitization"""
        # Compute statistics
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        
        # Detect and correct anomalies (3-sigma rule)
        z_scores = (x - mean) / (std + 1e-8)
        anomaly_mask = torch.abs(z_scores) > 3
        
        # Replace anomalies with mean
        x_sanitized = x.clone()
        x_sanitized[anomaly_mask] = mean.expand_as(x)[anomaly_mask]
        
        return x_sanitized

    def _smoothing_sanitization(self, x: torch.Tensor) -> torch.Tensor:
        """Smoothing-based input sanitization"""
        # Apply Gaussian smoothing
        kernel_size = self.config.get('smoothing_kernel_size', 3)
        sigma = self.config.get('smoothing_sigma', 1.0)
        
        # 1D Gaussian smoothing for each feature
        x_smoothed = F.gaussian_filter1d(x, kernel_size, sigma)
        
        return x_smoothed

class FeatureSqueezer(nn.Module):
    """Feature squeezing for adversarial defense"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.squeezing_methods = config.get('squeezing_methods', ['bit_reduction', 'smoothing'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature squeezing to input"""
        squeezed_outputs = []
        
        for method in self.squeezing_methods:
            if method == 'bit_reduction':
                squeezed = self._bit_reduction_squeezing(x)
            elif method == 'smoothing':
                squeezed = self._smoothing_squeezing(x)
            else:
                squeezed = x
            
            squeezed_outputs.append(squeezed)
        
        # Combine squeezed features
        if len(squeezed_outputs) > 1:
            # Use attention to combine different squeezed versions
            combined = self._attention_combination(squeezed_outputs)
        else:
            combined = squeezed_outputs[0]
        
        return combined

    def _bit_reduction_squeezing(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce bit depth of features"""
        bit_depth = self.config.get('bit_depth', 4)
        max_val = 2 ** bit_depth - 1
        
        # Scale to [0, max_val], round, then scale back
        x_scaled = x * max_val
        x_rounded = torch.round(x_scaled)
        x_reduced = x_rounded / max_val
        
        return x_reduced

    def _smoothing_squeezing(self, x: torch.Tensor) -> torch.Tensor:
        """Smoothing-based feature squeezing"""
        kernel_size = self.config.get('squeezing_kernel_size', 3)
        
        # Apply median filtering
        x_smoothed = F.avg_pool1d(x.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2)
        x_smoothed = x_smoothed.squeeze(1)
        
        return x_smoothed

    def _attention_combination(self, squeezed_outputs: list) -> torch.Tensor:
        """Combine squeezed features with attention"""
        # Simple average combination
        # In practice, you might use a learned attention mechanism
        stacked = torch.stack(squeezed_outputs, dim=0)
        combined = torch.mean(stacked, dim=0)
        
        return combined

class GradientMasker(nn.Module):
    """Gradient masking for adversarial defense"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient masking during inference"""
        if not self.training:
            # Add noise during inference to mask gradients
            noise_std = self.config.get('inference_noise_std', 0.01)
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        
        return x