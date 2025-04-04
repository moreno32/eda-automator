"""
Core module for EDA Automator

This module contains the central components for performing exploratory data analysis.
"""

# Import core components
from eda_automator.core.automator import EDACore

# Import submodules for easier access
from eda_automator.core import (
    analysis,
    data,
    visualization,
    report_generators,
    utils
)

__all__ = [
    'EDACore',
    'analysis',
    'data',
    'visualization',
    'report_generators',
    'utils'
] 