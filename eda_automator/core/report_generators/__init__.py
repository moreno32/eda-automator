"""
Report generators for EDA Automator

This module provides functions for generating reports in various formats (HTML, Markdown, etc.)
from the results of exploratory data analysis.
"""

from importlib import import_module
from typing import Dict, List, Any, Optional, Union

# Import report generators
from eda_automator.core.report_generators.html import generate_html_report
from eda_automator.core.report_generators.markdown import generate_markdown_report
from eda_automator.core.report_generators.image import generate_image_report

# Available report formats
AVAILABLE_FORMATS = {
    'html': generate_html_report,
    'md': generate_markdown_report,
    'markdown': generate_markdown_report,
    'image': generate_image_report,
}

def generate_report(
    output_path: str,
    format: str = 'html',
    data=None,
    results=None,
    figures=None,
    target_variable=None,
    settings=None,
    template_path=None,
    include_code=True,
    **kwargs
) -> str:
    """
    Generate a report from the EDA results.
    
    Parameters
    ----------
    output_path : str
        Path to save the report
    format : str, default 'html'
        Report format ('html', 'markdown', 'md', 'image')
    data : pandas.DataFrame, optional
        Data used for the analysis
    results : dict, optional
        Results from the analysis
    figures : dict, optional
        Visualization figures
    target_variable : str, optional
        Name of the target variable
    settings : dict, optional
        Settings for the report
    template_path : str, optional
        Path to a custom template file
    include_code : bool, default True
        Whether to include Python code snippets in the report
        
    Returns
    -------
    str
        Path to the generated report
    """
    # Check if format is valid
    if format not in AVAILABLE_FORMATS:
        raise ValueError(
            f"Invalid report format: {format}. "
            f"Available formats: {', '.join(AVAILABLE_FORMATS.keys())}"
        )
    
    # Get report generator function
    generator = AVAILABLE_FORMATS[format]
    
    # Generate report
    report_path = generator(
        output_path=output_path,
        data=data,
        results=results,
        figures=figures,
        target_variable=target_variable,
        settings=settings,
        template_path=template_path,
        include_code=include_code,
        **kwargs
    )
    
    return report_path

__all__ = [
    'generate_report',
    'generate_html_report',
    'generate_markdown_report',
    'generate_image_report',
    'AVAILABLE_FORMATS'
] 