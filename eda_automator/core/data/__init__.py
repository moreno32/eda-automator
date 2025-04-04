"""
Data module for EDA Automator

This module provides functions for loading, manipulating, and generating datasets.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

# Import data functions
from eda_automator.core.data.loader import load_data, load_from_url
from eda_automator.core.data.generator import (
    create_dataset,
    create_basic_dataset,
    create_timeseries_dataset,
    create_mixed_dataset
)

__all__ = [
    # Data loading
    'load_data',
    'load_from_url',
    
    # Dataset generation
    'create_dataset',
    'create_basic_dataset',
    'create_timeseries_dataset',
    'create_mixed_dataset'
] 