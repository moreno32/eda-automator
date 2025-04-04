"""
Environment setup utilities for EDA Automator

This module handles setting up the environment for EDA Automator,
including language, plotting settings, and other configuration.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import locale
from typing import Optional, Dict, Any

# Default settings
DEFAULT_SETTINGS = {
    'language': 'en',
    'theme': 'light',
    'palette': 'default',
    'figure_size': (10, 6),
    'font_scale': 1.0,
    'show_grid': True,
    'decimal_places': 2,
    'max_display_rows': 20,
    'sampling_threshold': 10000
}

# Available languages
AVAILABLE_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish'
}

# Translation dictionaries
TRANSLATIONS = {
    'en': {
        'missing_data': 'Missing Data',
        'distribution': 'Distribution',
        'correlation': 'Correlation',
        'outliers': 'Outliers',
        'summary_stats': 'Summary Statistics'
    },
    'es': {
        'missing_data': 'Datos Faltantes',
        'distribution': 'Distribución',
        'correlation': 'Correlación',
        'outliers': 'Valores Atípicos',
        'summary_stats': 'Estadísticas Resumidas'
    }
}

def setup_environment(
    language: str = 'en',
    theme: str = 'light',
    palette: str = 'default',
    figure_size: tuple = (10, 6),
    font_scale: float = 1.0,
    custom_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Configure the environment for EDA Automator.
    
    Parameters
    ----------
    language : str, default 'en'
        Language for reports and plots ('en' or 'es')
    theme : str, default 'light'
        Theme for plots ('light' or 'dark')
    palette : str, default 'default'
        Color palette for plots
    figure_size : tuple, default (10, 6)
        Default figure size for plots (width, height) in inches
    font_scale : float, default 1.0
        Scaling factor for font sizes
    custom_settings : dict, optional
        Additional custom settings
        
    Returns
    -------
    dict
        Complete settings dictionary
    """
    # Create settings dictionary
    settings = DEFAULT_SETTINGS.copy()
    
    # Update with provided settings
    settings.update({
        'language': language,
        'theme': theme,
        'palette': palette,
        'figure_size': figure_size,
        'font_scale': font_scale
    })
    
    # Update with any custom settings
    if custom_settings:
        settings.update(custom_settings)
    
    # Set locale based on language
    try:
        if language == 'es':
            locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
        else:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        # Fallback if locale not available
        locale.setlocale(locale.LC_ALL, '')
    
    # Configure pandas display
    pd.set_option('display.max_rows', settings['max_display_rows'])
    pd.set_option('display.precision', settings['decimal_places'])
    
    # Configure matplotlib
    plt.rcParams['figure.figsize'] = settings['figure_size']
    plt.rcParams['font.size'] = 10 * settings['font_scale']
    
    # Set theme
    if theme == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    return settings

def get_translation(key: str, language: str = 'en') -> str:
    """
    Get translated text for the given key and language.
    
    Parameters
    ----------
    key : str
        Translation key
    language : str, default 'en'
        Language code
        
    Returns
    -------
    str
        Translated text
    """
    if language not in TRANSLATIONS:
        language = 'en'
        
    translations = TRANSLATIONS[language]
    return translations.get(key, key) 