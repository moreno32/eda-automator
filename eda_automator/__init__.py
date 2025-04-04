"""
EDA Automator - Automated Exploratory Data Analysis package.

This package provides tools for automated exploratory data analysis (EDA).
"""

# Import core functionality
from eda_automator.core import EDACore
from eda_automator.core.utils import setup_environment
from eda_automator.core.data import load_data, create_dataset

# Keep backward compatibility
try:
    from eda_automator.unified import EDAAutomator
except ImportError:
    EDAAutomator = EDACore  # Fallback to core implementation

__all__ = [
    'EDACore',          # New main class
    'EDAAutomator',     # Legacy class (for backward compatibility)
    'setup_environment',
    'load_data',
    'create_dataset'
]

# Version
__version__ = '0.2.0' 