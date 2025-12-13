import torch
import numpy as np
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import hashlib
import time
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration management for CyberShieldNet"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        
    def load_config(self, config_name: str) -> Dict:
        """Load configuration from YAML file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.configs[config_name] = config
        logger.info(f"Loaded configuration: {config_name}")
        return config
    
    def save_config(self, config_name: str, config: Dict):
        """Save configuration to YAML file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.configs[config_name] = config
        logger.info(f"Saved configuration: {config_name}")
    
    def get_config(self, config_name: str, default: Optional[Dict] = None) -> Dict:
        """Get configuration, load if not already loaded"""
        if config_name not in self.configs:
            try:
                return self.load_config(config_name)
            except FileNotFoundError:
                if default is not None:
                    return default
                raise
        
        return self.configs[config_name]
    
    def update_config(self, config_name: str, updates: Dict):
        """Update configuration with new values"""
        config = self.get_config(config_name)
        
        def deep_update(source, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in source and isinstance(source[key], dict):
                    deep_update(source[key], value)
                else:
                    source[key] = value
        
        deep_update(config, updates)
        self.save_config(config_name, config)
        logger.info(f"Updated configuration: {config_name}")

class ModelCheckpoint:
    """Model checkpoint management"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict,
                       is_best: bool = False,
                       filename: Optional[str] = None) -> str:
        """Save model checkpoint"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
        """Load model checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint file"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        return str(latest_checkpoint)
    
    def cleanup_old_checkpoints(self, keep_n: int = 5):
        """Clean up old checkpoints, keeping only the most recent ones"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoint_files) <= keep_n:
            return
        
        # Sort by modification time and remove old ones
        sorted_checkpoints = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_checkpoint in sorted_checkpoints[keep_n:]:
            old_checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {old_checkpoint}")

class DataLoader:
    """Enhanced data loader with caching and preprocessing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, 
                 data_path: str,
                 use_cache: bool = True,
                 force_reload: bool = False) -> Any:
        """Load data with optional caching"""
        data_path = Path(data_path)
        cache_key = self._generate_cache_key(data_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache
        if use_cache and cache_file.exists() and not force_reload:
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded cached data: {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}")
        
        # Load original data
        data = self._load_raw_data(data_path)
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Cache processed data
        if use_cache:
            try:
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                logger.info(f"Cached processed data: {cache_file}")
            except Exception as e:
                logger.warning(f"Caching failed: {e}")
        
        return processed_data
    
    def _generate_cache_key(self, data_path: Path) -> str:
        """Generate cache key based on file path and modification time"""
        stat = data_path.stat()
        key_data = f"{data_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_raw_data(self, data_path: Path) -> Any:
        """Load raw data based on file extension"""
        suffix = data_path.suffix.lower()
        
        if suffix == '.json':
            with open(data_path, 'r') as f:
                return json.load(f)
        elif suffix in ['.pkl', '.pickle']:
            import pickle
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        elif suffix == '.npy':
            return np.load(data_path)
        elif suffix == '.pt':
            return torch.load(data_path)
        elif suffix in ['.csv', '.tsv']:
            import pandas as pd
            return pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _preprocess_data(self, data: Any) -> Any:
        """Preprocess loaded data"""
        # Implement data-specific preprocessing
        if isinstance(data, dict):
            return self._preprocess_dict_data(data)
        elif isinstance(data, (list, tuple)):
            return self._preprocess_sequence_data(data)
        elif isinstance(data, np.ndarray):
            return self._preprocess_array_data(data)
        else:
            return data
    
    def _preprocess_dict_data(self, data: Dict) -> Dict:
        """Preprocess dictionary data"""
        processed = {}
        for key, value in data.items():
            if isinstance(value, (dict, list, np.ndarray)):
                processed[key] = self._preprocess_data(value)
            else:
                processed[key] = value
        return processed
    
    def _preprocess_sequence_data(self, data: List) -> List:
        """Preprocess sequence data"""
        return [self._preprocess_data(item) for item in data]
    
    def _preprocess_array_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess array data"""
        # Normalize and handle missing values
        if np.issubdtype(data.dtype, np.number):
            # Replace NaN with mean
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=np.nanmean(data))
            
            # Normalize if configured
            if self.config.get('normalize', True):
                data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        return data

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.metrics_history = {}
        self.start_time = time.time()
        self.batch_times = []
    
    def start_batch_timer(self):
        """Start timing a batch"""
        self.batch_start = time.time()
    
    def end_batch_timer(self):
        """End timing a batch and record duration"""
        if hasattr(self, 'batch_start'):
            batch_time = time.time() - self.batch_start
            self.batch_times.append(batch_time)
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append(value)
    
    def get_throughput(self, batch_size: int) -> float:
        """Calculate current throughput (samples per second)"""
        if not self.batch_times:
            return 0.0
        
        avg_batch_time = np.mean(self.batch_times[-10:])  # Last 10 batches
        return batch_size / avg_batch_time if avg_batch_time > 0 else 0.0
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage statistics"""
        import psutil
        process = psutil.Process()
        
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_rss_mb': memory_info.rss / 1024 / 1024,
            'process_vms_mb': memory_info.vms / 1024 / 1024,
            'system_available_mb': system_memory.available / 1024 / 1024,
            'system_used_percent': system_memory.percent
        }
    
    def get_gpu_usage(self) -> Dict:
        """Get GPU usage statistics"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_memory_max = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                
                return {
                    'gpu_memory_used_mb': gpu_memory,
                    'gpu_memory_max_mb': gpu_memory_max,
                    'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                }
        except Exception as e:
            logger.warning(f"GPU monitoring failed: {e}")
        
        return {}
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'uptime_seconds': time.time() - self.start_time,
            'total_batches_processed': len(self.batch_times),
            'average_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'batch_time_std': np.std(self.batch_times) if self.batch_times else 0,
            'metrics_tracked': list(self.metrics_history.keys()),
            'memory_usage': self.get_memory_usage(),
            'gpu_usage': self.get_gpu_usage()
        }
        
        # Add latest metric values
        for metric_name, values in self.metrics_history.items():
            if values:
                report[f'{metric_name}_current'] = values[-1]
                report[f'{metric_name}_average'] = np.mean(values)
                report[f'{metric_name}_trend'] = self._calculate_trend(values)
        
        return report
    
    def _calculate_trend(self, values: List[float], window: int = 10) -> str:
        """Calculate trend of metric values"""
        if len(values) < 2:
            return "stable"
        
        recent_values = values[-window:]
        if len(recent_values) < 2:
            return "stable"
        
        # Simple linear trend
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "deteriorating"
        else:
            return "stable"

class SecurityUtils:
    """Security-related utility functions"""
    
    @staticmethod
    def sanitize_input(data: Any) -> Any:
        """Sanitize input data to prevent injection attacks"""
        if isinstance(data, str):
            # Remove potentially dangerous characters
            import re
            data = re.sub(r'[<>&\"\']', '', data)
        elif isinstance(data, dict):
            return {k: SecurityUtils.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [SecurityUtils.sanitize_input(item) for item in data]
        
        return data
    
    @staticmethod
    def validate_model_file(file_path: str) -> bool:
        """Validate model file integrity"""
        try:
            # Check file signature and structure
            checkpoint = torch.load(file_path, map_location='cpu')
            
            required_keys = ['model_state_dict', 'config']
            if not all(key in checkpoint for key in required_keys):
                return False
            
            # Additional validation can be added here
            return True
            
        except Exception as e:
            logger.error(f"Model file validation failed: {e}")
            return False
    
    @staticmethod
    def generate_secure_random(seed: Optional[int] = None) -> int:
        """Generate cryptographically secure random number"""
        if seed is not None:
            random.seed(seed)
        
        # Use system random for better security
        import secrets
        return secrets.randbelow(2**32 - 1)
    
    @staticmethod
    def encrypt_sensitive_data(data: str, key: str) -> str:
        """Encrypt sensitive data (simplified version)"""
        # In production, use proper encryption libraries
        import base64
        from cryptography.fernet import Fernet
        
        # Generate key from password
        key_base = base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest())
        fernet = Fernet(key_base)
        
        encrypted = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    @staticmethod
    def decrypt_sensitive_data(encrypted_data: str, key: str) -> str:
        """Decrypt sensitive data (simplified version)"""
        import base64
        from cryptography.fernet import Fernet
        
        try:
            key_base = base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest())
            fernet = Fernet(key_base)
            
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

class VisualizationUtils:
    """Utility functions for data visualization"""
    
    @staticmethod
    def plot_training_history(history: Dict, save_path: Optional[str] = None):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        if 'train_loss' in history:
            plt.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        
        # Plot accuracy if available
        plt.subplot(1, 2, 2)
        if 'train_accuracy' in history:
            plt.plot(history['train_accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def create_confusion_matrix_plot(y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   class_names: List[str],
                                   save_path: Optional[str] = None):
        """Create confusion matrix visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_importance: Dict, 
                              top_n: int = 15,
                              save_path: Optional[str] = None):
        """Plot feature importance"""
        import matplotlib.pyplot as plt
        
        # Get top N features
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importance, align='center', alpha=0.7)
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()