"""
Configuration module for the unified EDA reports.

This module contains constants and configuration settings for the
unified EDA reports application.
"""

import os
import locale
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# Output directory
DEFAULT_OUTPUT_DIR = os.path.join("output", "unified")

# Analysis thresholds
DEFAULT_QUALITY_THRESHOLD = 7.0
DEFAULT_MISSING_THRESHOLD = 10.0
DEFAULT_CORRELATION_THRESHOLD = 0.7
DEFAULT_OUTLIER_THRESHOLD = 0.05

# Visualization settings
MAX_CATEGORIES_FOR_BAR = 15
MAX_CATEGORIES_FOR_PIE = 10
MAX_CATEGORIES_FOR_HEATMAP = 25
FIGURE_SIZE = (10, 6)
FIGURE_DPI = 100

# Report settings
HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "report_template.html")
CSS_STYLE_PATH = os.path.join(os.path.dirname(__file__), "templates", "report_style.css")

# Data type mappings
DATA_TYPE_MAPPINGS = {
    "numerical": ["int64", "float64", "int32", "float32"],
    "categorical": ["object", "category", "bool"],
    "datetime": ["datetime64", "datetime64[ns]"]
}

# File extensions for supported formats
SUPPORTED_FILE_EXTENSIONS = {
    "csv": [".csv"],
    "excel": [".xlsx", ".xls", ".xlsm"],
    "parquet": [".parquet"],
    "json": [".json"],
    "pickle": [".pkl", ".pickle"]
}

# Display settings
DISPLAY_MAX_ROWS = 100
DISPLAY_MAX_COLUMNS = 50
DISPLAY_WIDTH = 1000

def setup_environment(language='en', suppress_warnings=True):
    """
    Configure global environment settings for reports.
    
    Args:
        language (str): Language code ('en' for English, 'es' for Spanish)
        suppress_warnings (bool): Whether to suppress warnings
    """
    # Set locale for number and date formatting
    try:
        if language == 'es':
            locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
        else:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        print(f"Warning: Could not set locale for {language}. Using system default.")
    
    # Matplotlib configuration
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['figure.dpi'] = FIGURE_DPI
    
    # Configure pandas display options
    pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)
    pd.set_option('display.max_columns', DISPLAY_MAX_COLUMNS)
    pd.set_option('display.width', DISPLAY_WIDTH)
    
    # Suppress warnings if requested
    if suppress_warnings:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
    
    print(f"Environment setup complete. Language: {language}") 