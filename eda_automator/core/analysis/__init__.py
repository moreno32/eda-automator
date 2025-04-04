"""
Analysis module for EDA Automator

This module provides functions for performing various types of exploratory data analysis.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

# Import analysis functions
from eda_automator.core.analysis.basic import run_basic_analysis
from eda_automator.core.analysis.missing import run_missing_analysis
from eda_automator.core.analysis.outliers import run_outlier_analysis
from eda_automator.core.analysis.correlation import run_correlation_analysis
from eda_automator.core.analysis.distribution import run_distribution_analysis
from eda_automator.core.analysis.target import run_target_analysis

def run_full_analysis(
    data: pd.DataFrame,
    target_variable: Optional[str] = None,
    analyses: Optional[List[str]] = None,
    settings: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Run a full exploratory data analysis on the provided DataFrame.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to analyze
    target_variable : str, optional
        Name of the target variable for supervised analysis
    analyses : list of str, optional
        List of analyses to perform. If None, all analyses will be performed.
        Options: 'basic', 'missing', 'outliers', 'correlation', 'distribution', 'target'
    settings : dict, optional
        Settings for the analysis
        
    Returns
    -------
    dict
        Results of the analysis
    """
    if settings is None:
        settings = {}
    
    if analyses is None:
        analyses = ['basic', 'missing', 'outliers', 'correlation', 'distribution']
        if target_variable is not None:
            analyses.append('target')
    
    results = {}
    
    # Run basic analysis
    if 'basic' in analyses:
        results['basic'] = run_basic_analysis(data, settings.get('basic', {}))
    
    # Run missing values analysis
    if 'missing' in analyses:
        results['missing_values'] = run_missing_analysis(data, settings.get('missing', {}))
    
    # Run outlier analysis
    if 'outliers' in analyses:
        results['outliers'] = run_outlier_analysis(data, settings.get('outliers', {}))
    
    # Run correlation analysis
    if 'correlation' in analyses:
        results['correlation'] = run_correlation_analysis(data, settings.get('correlation', {}))
    
    # Run distribution analysis
    if 'distribution' in analyses:
        results['distribution'] = run_distribution_analysis(data, settings.get('distribution', {}))
    
    # Run target analysis
    if 'target' in analyses and target_variable is not None:
        results['target_analysis'] = run_target_analysis(
            data, target_variable, settings.get('target', {})
        )
    
    return results

__all__ = [
    # Main functions
    'run_full_analysis',
    
    # Individual analysis functions
    'run_basic_analysis',
    'run_missing_analysis',
    'run_outlier_analysis',
    'run_correlation_analysis',
    'run_distribution_analysis',
    'run_target_analysis'
] 