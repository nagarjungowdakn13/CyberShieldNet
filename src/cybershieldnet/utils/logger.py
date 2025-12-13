import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

def setup_logger(name: str, 
                log_level: str = "INFO",
                log_file: Optional[str] = None,
                console_output: bool = True) -> logging.Logger:
    """
    Setup logger with consistent configuration
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set log level
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

class PerformanceLogger:
    """
    Performance logging for model training and evaluation
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Training log file
        self.training_log = self.log_dir / "training_performance.jsonl"
        self.evaluation_log = self.log_dir / "evaluation_performance.jsonl"
        
    def log_training_epoch(self, 
                         epoch: int,
                         train_loss: float,
                         val_loss: Optional[float] = None,
                         metrics: Optional[Dict] = None,
                         timestamp: Optional[str] = None):
        """Log training epoch performance"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics or {}
        }
        
        with open(self.training_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_evaluation(self,
                     model_name: str,
                     dataset_name: str,
                     metrics: Dict,
                     timestamp: Optional[str] = None):
        """Log evaluation performance"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'model': model_name,
            'dataset': dataset_name,
            'metrics': metrics
        }
        
        with open(self.evaluation_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_training_history(self) -> list:
        """Get training history from log file"""
        if not self.training_log.exists():
            return []
        
        history = []
        with open(self.training_log, 'r') as f:
            for line in f:
                history.append(json.loads(line.strip()))
        
        return history
    
    def get_evaluation_history(self) -> list:
        """Get evaluation history from log file"""
        if not self.evaluation_log.exists():
            return []
        
        history = []
        with open(self.evaluation_log, 'r') as f:
            for line in f:
                history.append(json.loads(line.strip()))
        
        return history

class SecurityLogger:
    """
    Security-focused logging for threat detection events
    """
    
    def __init__(self, log_dir: str = "logs/security"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.threat_log = self.log_dir / "threat_events.jsonl"
        self.incident_log = self.log_dir / "security_incidents.jsonl"
        
    def log_threat_detection(self,
                           threat_type: str,
                           confidence: float,
                           assets_affected: List[str],
                           risk_score: float,
                           timestamp: Optional[str] = None):
        """Log threat detection event"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'threat_detection',
            'threat_type': threat_type,
            'confidence': confidence,
            'assets_affected': assets_affected,
            'risk_score': risk_score
        }
        
        with open(self.threat_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_security_incident(self,
                            incident_type: str,
                            severity: str,
                            description: str,
                            mitigation_action: str,
                            timestamp: Optional[str] = None):
        """Log security incident"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'security_incident',
            'incident_type': incident_type,
            'severity': severity,
            'description': description,
            'mitigation_action': mitigation_action
        }
        
        with open(self.incident_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_recent_threats(self, hours: int = 24) -> list:
        """Get recent threat events within specified hours"""
        if not self.threat_log.exists():
            return []
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_threats = []
        
        with open(self.threat_log, 'r') as f:
            for line in f:
                event = json.loads(line.strip())
                event_time = datetime.fromisoformat(event['timestamp']).timestamp()
                if event_time >= cutoff_time:
                    recent_threats.append(event)
        
        return recent_threats

class AuditLogger:
    """
    Audit logging for model changes and system operations
    """
    
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.audit_log = self.log_dir / "system_audit.jsonl"
        
    def log_model_update(self,
                       model_name: str,
                       old_version: str,
                       new_version: str,
                       performance_change: Dict,
                       user: str = "system"):
        """Log model update event"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'model_update',
            'model_name': model_name,
            'old_version': old_version,
            'new_version': new_version,
            'performance_change': performance_change,
            'user': user
        }
        
        with open(self.audit_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_configuration_change(self,
                               component: str,
                               old_config: Dict,
                               new_config: Dict,
                               user: str = "system"):
        """Log configuration change"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'configuration_change',
            'component': component,
            'old_config': old_config,
            'new_config': new_config,
            'user': user
        }
        
        with open(self.audit_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_security_event(self,
                         event_type: str,
                         description: str,
                         severity: str,
                         user: str = "system"):
        """Log security-related event"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'security_event',
            'event_subtype': event_type,
            'description': description,
            'severity': severity,
            'user': user
        }
        
        with open(self.audit_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')