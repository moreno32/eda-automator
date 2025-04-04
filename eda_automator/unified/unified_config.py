"""
Configuration module for EDA Automator.

This module contains configuration parameters and settings for the EDA Automator package.
"""

import os
import locale
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# Output directory
DEFAULT_OUTPUT_DIR = os.path.join("output", "unified")

# Data validation parameters
MIN_ROWS = 10
MAX_ROWS = 1000000
MAX_COLS = 1000
MAX_MEMORY_MB = 1000

# Missing data threshold (percentage above which to flag a variable)
MISSING_THRESHOLD = 0.5

# Outlier detection parameters
ZSCORE_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.5

# Correlation parameters
CORRELATION_THRESHOLD = 0.7

# Visualization parameters
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

# Reporting parameters
MAX_KEY_FINDINGS = 10
MAX_PLOTS_PER_CATEGORY = 20

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