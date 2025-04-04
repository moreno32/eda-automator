"""
Dependencies checking utilities

This module checks for optional dependencies required by EDA Automator.
"""

import importlib
import subprocess
import os
import sys
from typing import Dict, Any, List, Tuple

def check_dependencies() -> Dict[str, bool]:
    """
    Check for all optional dependencies required by EDA Automator.
    
    Returns
    -------
    dict
        Dictionary of dependency names and their availability status
    """
    dependencies = {
        # Data processing
        'pandas': is_module_available('pandas'),
        'numpy': is_module_available('numpy'),
        'scipy': is_module_available('scipy'),
        
        # Visualization
        'matplotlib': is_module_available('matplotlib'),
        'seaborn': is_module_available('seaborn'),
        'plotly': is_module_available('plotly'),
        
        # Machine learning
        'sklearn': is_module_available('sklearn'),
        
        # Report generation
        'jinja2': is_module_available('jinja2'),
        'markdown': is_module_available('markdown'),
        'excel': is_module_available('openpyxl'),
        
        # Image report generation
        'imgkit': is_module_available('imgkit'),
        'selenium': is_module_available('selenium'),
        'weasyprint': is_module_available('weasyprint'),
        'pdf2image': is_module_available('pdf2image'),
        
        # External tools
        'wkhtmltopdf': is_external_tool_available('wkhtmltopdf')
    }
    
    return dependencies

def is_module_available(module_name: str) -> bool:
    """
    Check if a Python module is available.
    
    Parameters
    ----------
    module_name : str
        Name of the module to check
        
    Returns
    -------
    bool
        True if the module is available, False otherwise
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def is_external_tool_available(tool_name: str) -> bool:
    """
    Check if an external tool is available in the system PATH.
    
    Parameters
    ----------
    tool_name : str
        Name of the tool to check
        
    Returns
    -------
    bool
        True if the tool is available, False otherwise
    """
    try:
        # Use 'where' on Windows, 'which' on Unix-like systems
        if os.name == 'nt':  # Windows
            result = subprocess.run(['where', tool_name], 
                                    capture_output=True, 
                                    text=True, 
                                    check=False)
        else:  # Unix-like
            result = subprocess.run(['which', tool_name], 
                                    capture_output=True, 
                                    text=True, 
                                    check=False)
            
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_missing_dependencies(required_deps: List[str]) -> List[str]:
    """
    Get a list of missing dependencies from a list of required ones.
    
    Parameters
    ----------
    required_deps : list
        List of required dependency names
        
    Returns
    -------
    list
        List of missing dependency names
    """
    deps = check_dependencies()
    return [dep for dep in required_deps if dep in deps and not deps[dep]]

def get_dependency_groups() -> Dict[str, List[Tuple[str, str]]]:
    """
    Get groups of dependencies with their descriptions.
    
    Returns
    -------
    dict
        Dictionary of dependency group names and lists of (name, description) tuples
    """
    return {
        'core': [
            ('pandas', 'Data manipulation library'),
            ('numpy', 'Numerical computing library'),
            ('matplotlib', 'Visualization library')
        ],
        'advanced_analysis': [
            ('scipy', 'Scientific computing library'),
            ('sklearn', 'Machine learning library')
        ],
        'enhanced_visualization': [
            ('seaborn', 'Statistical visualization library'),
            ('plotly', 'Interactive visualization library')
        ],
        'report_generation': [
            ('jinja2', 'Template engine for HTML reports'),
            ('markdown', 'Markdown report generation'),
            ('openpyxl', 'Excel report generation')
        ],
        'image_generation': [
            ('imgkit', 'HTML to image conversion (requires wkhtmltopdf)'),
            ('selenium', 'Web automation for screenshots'),
            ('weasyprint', 'HTML to PDF conversion'),
            ('pdf2image', 'PDF to image conversion')
        ]
    } 