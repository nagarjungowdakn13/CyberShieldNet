"""Shared constants for CyberShieldNet (compatibility shim).

This module exposes common names expected elsewhere in the codebase
such as `THREAT_CATEGORIES` and `DATA_MODALITIES`. It imports the
primary enums from the existing `constatnts.py` file to avoid
duplicating definitions while providing a correctly named module
(`constants`) for imports.
"""

from .constatnts import ThreatLevel, DataModality, ModelType, DEFAULT_CONFIG, PERFORMANCE_TARGETS

# Common threat categories (simplified list)
THREAT_CATEGORIES = [
    'apt_campaign', 'ransomware', 'insider_threat', 'ddos_attack',
    'zero_day_exploit', 'phishing_campaign', 'data_exfiltration',
    'lateral_movement', 'privilege_escalation', 'credential_theft'
]

# Common asset / data modalities exposed to API
DATA_MODALITIES = [
    DataModality.TEMPORAL.value,
    DataModality.STRUCTURAL.value,
    DataModality.BEHAVIORAL.value,
    DataModality.CONTEXTUAL.value,
    DataModality.THREAT_INTEL.value,
    DataModality.ASSET_CRITICALITY.value
]

__all__ = [
    'ThreatLevel', 'DataModality', 'ModelType',
    'DEFAULT_CONFIG', 'PERFORMANCE_TARGETS',
    'THREAT_CATEGORIES', 'DATA_MODALITIES'
]
