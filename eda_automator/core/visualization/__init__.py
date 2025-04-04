"""
Visualization module for EDA Automator

This module provides functions for creating visualizations of data and analysis results.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

# Import visualization functions
from eda_automator.core.visualization.basic import (
    plot_basic_info,
    plot_missing_values,
    plot_outliers
)

from eda_automator.core.visualization.distribution import (
    plot_distribution,
    plot_histograms,
    plot_boxplots,
    plot_categorical
)

from eda_automator.core.visualization.correlation import (
    plot_correlation_matrix,
    plot_feature_correlations,
    plot_scatter_matrix
)

from eda_automator.core.visualization.target import (
    plot_target_distribution,
    plot_feature_importance,
    plot_bivariate_analysis
)

def generate_all_figures(
    data: pd.DataFrame,
    results: Dict[str, Any],
    target_variable: Optional[str] = None,
    settings: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate all visualizations based on the analysis results.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data used for the analysis
    results : dict
        Results from the analysis
    target_variable : str, optional
        Name of the target variable
    settings : dict, optional
        Settings for the visualizations
        
    Returns
    -------
    dict
        Dictionary of visualization figures
    """
    if settings is None:
        settings = {}
    
    figures = {}
    
    # Generate basic figures if basic analysis results exist
    if 'basic' in results:
        figures.update(plot_basic_info(data, results['basic']))
    
    # Generate missing values figures if missing values analysis results exist
    if 'missing_values' in results:
        figures.update(plot_missing_values(data, results['missing_values']))
    
    # Generate outlier figures if outlier analysis results exist
    if 'outliers' in results:
        figures.update(plot_outliers(data, results['outliers']))
    
    # Generate distribution figures if distribution analysis results exist
    if 'distribution' in results:
        figures.update(plot_distribution(data, results['distribution']))
    
    # Generate correlation figures if correlation analysis results exist
    if 'correlation' in results:
        figures.update(plot_correlation_matrix(data, results['correlation']))
        figures.update(plot_feature_correlations(data, results['correlation']))
    
    # Generate target figures if target analysis results exist and target variable is specified
    if 'target_analysis' in results and target_variable is not None:
        figures.update(plot_target_distribution(data, results['target_analysis'], target_variable))
        figures.update(plot_feature_importance(data, results['target_analysis'], target_variable))
        figures.update(plot_bivariate_analysis(data, results['target_analysis'], target_variable))
    
    return figures

__all__ = [
    # Main function
    'generate_all_figures',
    
    # Basic visualization
    'plot_basic_info',
    'plot_missing_values',
    'plot_outliers',
    
    # Distribution visualization
    'plot_distribution',
    'plot_histograms',
    'plot_boxplots',
    'plot_categorical',
    
    # Correlation visualization
    'plot_correlation_matrix',
    'plot_feature_correlations',
    'plot_scatter_matrix',
    
    # Target visualization
    'plot_target_distribution',
    'plot_feature_importance',
    'plot_bivariate_analysis'
] 