"""CyberShieldNet: Multi-Modal Fusion Framework for Predictive Threat Intelligence.

This package exposes the main public symbols lazily so importing the
top-level package does not require heavy ML dependencies (like
PyTorch) to be installed. Accessing the actual classes will import
their modules on-demand and raise clear errors if required libraries
are missing.
"""

__version__ = "1.0.0"
__author__ = "Nagarjun Gowda K N"
__email__ = "nagarjun@gmail.com"

# Public API names (exported lazily)
__all__ = [
    "CyberShieldNet",
    "TemporalGraphFusion",
    "DynamicRiskPropagation",
]


def __getattr__(name: str):
    # Lazy-load heavy submodules only when requested
    if name == "CyberShieldNet":
        from .core.base_model import CyberShieldNet
        return CyberShieldNet
    if name == "TemporalGraphFusion":
        from .fusion_engine.tgcf import TemporalGraphFusion
        return TemporalGraphFusion
    if name == "DynamicRiskPropagation":
        from .risk_assessment.drpa import DynamicRiskPropagation
        return DynamicRiskPropagation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
