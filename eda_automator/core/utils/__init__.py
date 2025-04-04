"""
Utility functions for EDA Automator

This module provides common utility functions used across the EDA Automator package.
"""

from .environment import setup_environment, get_language, set_language
from .dependencies import check_dependencies, get_missing_dependencies
from .logger import setup_logging, get_logger
from .formatting import format_number, format_percent, format_list, format_dict

__all__ = [
    'setup_environment',
    'get_language',
    'set_language',
    'check_dependencies',
    'get_missing_dependencies',
    'setup_logging',
    'get_logger',
    'format_number',
    'format_percent',
    'format_list',
    'format_dict'
] 