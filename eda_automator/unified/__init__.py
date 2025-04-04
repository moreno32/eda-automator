"""
Módulo unificado para EDA Automator.

Este módulo centraliza todas las funcionalidades del análisis de datos exploratorio
en un solo lugar, incluyendo generadores de datos, análisis, visualizaciones y reportes.
"""

from .config import setup_environment
from .data import load_data, create_dataset
from .analysis import run_analysis
from .utils import setup_logging, generate_alerts_html, generate_recommendations
from .visualizations import generate_basic_visualizations, generate_univariate_plots, generate_bivariate_plots

__all__ = [
    'setup_environment',
    'load_data',
    'create_dataset',
    'run_analysis',
    'setup_logging',
    'generate_alerts_html',
    'generate_recommendations',
    'generate_basic_visualizations',
    'generate_univariate_plots',
    'generate_bivariate_plots'
] 