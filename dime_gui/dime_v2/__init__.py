"""
Dime_v2 - Industrial Anomaly Detection Library
"""

__version__ = "2.0.0"
__author__ = "Kundan kumar"
__email__ = "kundan.kumar@emagegroup.com"

from .api import DimeAnomalyDetector, create_detector

# Export main classes
__all__ = [
    'DimeAnomalyDetector',
    'create_detector'
]

# Export version
__version__ = "2.0.0"

# Optional: Export specific classes from core for advanced users
try:
    from .core.detector import DimeDetector
    from .enhanced_inference import EnhancedAnomalyInference
    from .inference import AnomalyInference
    __all__.extend(['DimeDetector', 'EnhancedAnomalyInference', 'AnomalyInference'])
except ImportError:
    pass