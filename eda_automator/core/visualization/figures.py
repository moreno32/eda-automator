"""
Main figure generation module for EDA Automator

This module provides the main entry point for generating visualizations from analysis results.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def generate_figures(
    df: pd.DataFrame,
    category: str,
    results: Dict[str, Any],
    target_variable: Optional[str] = None,
    max_figures: int = 20
) -> Dict[str, Any]:
    """
    Generate figures for a specific analysis category.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data
    category : str
        Analysis category ('basic', 'missing_values', 'outliers', 'correlation',
        'distribution', 'target_analysis')
    results : dict
        Analysis results for the specified category
    target_variable : str, optional
        Name of the target variable
    max_figures : int, default 20
        Maximum number of figures to generate
        
    Returns
    -------
    dict
        Dictionary of figure objects
    """
    logger.info(f"Generating figures for {category} analysis")
    
    # Import visualization modules
    from . import (
        plot_basic_info,
        plot_distributions,
        plot_correlation_matrix,
        plot_missing_values,
        plot_outliers,
        plot_target_relationships
    )
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        # Use non-interactive backend
        matplotlib.use('Agg')
    except ImportError:
        logger.error("matplotlib is required for visualization")
        return {"error": "matplotlib is required for visualization"}
    
    # Initialize figures dictionary
    figures = {}
    
    # Generate figures based on category
    if category == 'basic':
        figures = plot_basic_info(df, results)
    
    elif category == 'missing_values':
        figures = plot_missing_values(df, results)
    
    elif category == 'outliers':
        figures = plot_outliers(df, results)
    
    elif category == 'correlation':
        figures = plot_correlation_matrix(df, results)
    
    elif category == 'distribution':
        figures = plot_distributions(df, results, max_figures=max_figures)
    
    elif category == 'target_analysis':
        if target_variable is not None:
            figures = plot_target_relationships(df, results, target_variable)
    
    else:
        logger.warning(f"Unknown analysis category: {category}")
    
    # Limit number of figures if needed
    if len(figures) > max_figures:
        logger.info(f"Limiting to {max_figures} figures")
        figure_keys = list(figures.keys())
        for key in figure_keys[max_figures:]:
            del figures[key]
    
    # Convert figures to format suitable for serialization
    serialized_figures = {}
    
    for key, fig in figures.items():
        try:
            # Save figure to bytes
            import io
            from matplotlib.figure import Figure
            
            if isinstance(fig, Figure):
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                
                # Store serialized figure
                serialized_figures[key] = {
                    'type': 'matplotlib',
                    'format': 'png',
                    'bytes': buf.getvalue(),
                    'title': fig._suptitle.get_text() if fig._suptitle else key
                }
                
                # Close figure to free memory
                plt.close(fig)
            else:
                # For non-matplotlib figures
                serialized_figures[key] = fig
        except Exception as e:
            logger.warning(f"Error serializing figure {key}: {str(e)}")
    
    logger.info(f"Generated {len(serialized_figures)} figures for {category} analysis")
    return serialized_figures

def save_figure(
    fig,
    filename: str,
    dpi: int = 300,
    format: str = 'png'
) -> None:
    """
    Save a figure to a file.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Path to save the figure
    dpi : int, default 300
        Resolution in dots per inch
    format : str, default 'png'
        File format ('png', 'pdf', 'svg', 'jpg')
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        if isinstance(fig, Figure):
            fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
            logger.info(f"Saved figure to {filename}")
        else:
            logger.warning(f"Cannot save non-matplotlib figure to {filename}")
    except Exception as e:
        logger.error(f"Error saving figure to {filename}: {str(e)}")

def set_plot_style():
    """
    Set the default style for matplotlib plots.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set larger font sizes
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 14
        
        # Set color cycle
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ])
        
        # Other settings
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        logger.info("Set matplotlib style for EDA Automator")
    except ImportError:
        logger.warning("matplotlib not available, cannot set plot style")
    except Exception as e:
        logger.warning(f"Error setting plot style: {str(e)}") 