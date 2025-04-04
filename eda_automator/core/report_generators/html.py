"""
HTML report generation module for EDA Automator

This module provides functions for generating HTML reports from analysis results.
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List

from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def generate_html_report(
    output_path: str,
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
    Generate an HTML report from the EDA results.
    
    Parameters
    ----------
    output_path : str
        Path to save the HTML report
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
        Path to a custom HTML template file
    include_code : bool, default True
        Whether to include Python code snippets in the report
        
    Returns
    -------
    str
        Path to the generated HTML report
    """
    logger.info(f"Generating HTML report to {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Get default template if not provided
    if template_path is None:
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'report_template.html')
        
        # If default template doesn't exist, use basic template
        if not os.path.exists(template_path):
            logger.warning("Default template not found, using basic template")
            html_content = _generate_basic_html(data, results, figures, target_variable, settings)
            
            # Write HTML to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Generated HTML report at {output_path}")
            return output_path
    
    # Load template
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except Exception as e:
        logger.error(f"Error loading template: {str(e)}")
        logger.warning("Using basic template")
        html_content = _generate_basic_html(data, results, figures, target_variable, settings)
        
        # Write HTML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Generated HTML report at {output_path}")
        return output_path
    
    # Generate report content
    try:
        # Get report content as HTML sections
        sections_html = {}
        
        # Add basic info section
        if results and 'basic' in results:
            sections_html['basic_info'] = _generate_basic_info_section(results['basic'], data)
        
        # Add missing values section
        if results and 'missing_values' in results:
            sections_html['missing_values'] = _generate_missing_values_section(results['missing_values'])
        
        # Add outliers section
        if results and 'outliers' in results:
            sections_html['outliers'] = _generate_outliers_section(results['outliers'])
        
        # Add correlation section
        if results and 'correlation' in results:
            sections_html['correlation'] = _generate_correlation_section(results['correlation'])
        
        # Add distribution section
        if results and 'distribution' in results:
            sections_html['distribution'] = _generate_distribution_section(results['distribution'])
        
        # Add target analysis section
        if results and 'target_analysis' in results:
            sections_html['target_analysis'] = _generate_target_analysis_section(
                results['target_analysis'],
                target_variable
            )
        
        # Include figures
        figures_html = _embed_figures(figures)
        
        # Include code snippets if requested
        code_snippets = {}
        if include_code:
            code_snippets = _generate_code_snippets(results, target_variable)
        
        # Fill template
        html_content = template
        
        # Replace placeholders
        for section_name, content in sections_html.items():
            placeholder = f'{{{{ {section_name} }}}}'
            html_content = html_content.replace(placeholder, content)
        
        # Replace figures
        figures_placeholder = '{{ figures }}'
        html_content = html_content.replace(figures_placeholder, figures_html)
        
        # Replace code snippets
        code_placeholder = '{{ code_snippets }}'
        code_html = '\n'.join(code_snippets.values())
        html_content = html_content.replace(code_placeholder, code_html)
        
        # Replace metadata
        html_content = html_content.replace('{{ title }}', 'EDA Automator Report')
        html_content = html_content.replace('{{ date }}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        if data is not None:
            data_info = f"DataFrame with {data.shape[0]} rows and {data.shape[1]} columns"
        else:
            data_info = "No data provided"
        
        html_content = html_content.replace('{{ data_info }}', data_info)
        
        if target_variable:
            html_content = html_content.replace('{{ target_variable }}', f"Target variable: {target_variable}")
        else:
            html_content = html_content.replace('{{ target_variable }}', "No target variable specified")
    
    except Exception as e:
        logger.error(f"Error generating report content: {str(e)}")
        logger.warning("Using basic template")
        html_content = _generate_basic_html(data, results, figures, target_variable, settings)
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    logger.info(f"Generated HTML report at {output_path}")
    return output_path

def _generate_basic_html(
    data=None,
    results=None,
    figures=None,
    target_variable=None,
    settings=None
) -> str:
    """
    Generate a basic HTML report when no template is available.
    
    Parameters
    ----------
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
        
    Returns
    -------
    str
        HTML content
    """
    # Create basic HTML structure
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Automator Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
        h1, h2, h3, h4 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        .figure {{ margin: 20px 0; text-align: center; }}
        .figure img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metadata {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
        .code {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>EDA Automator Report</h1>
        <div class="metadata">
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    # Add data info
    if data is not None:
        html += f"            <p>DataFrame with {data.shape[0]} rows and {data.shape[1]} columns</p>\n"
    
    # Add target variable info
    if target_variable:
        html += f"            <p>Target variable: {target_variable}</p>\n"
    
    html += """        </div>
"""
    
    # Add results summary
    if results:
        html += """        <div class="section">
            <h2>Analysis Summary</h2>
            <ul>
"""
        
        for category, result in results.items():
            if result:
                html += f"                <li>{category.replace('_', ' ').title()}: Analysis completed</li>\n"
        
        html += """            </ul>
        </div>
"""
    
    # Add figures
    if figures:
        html += """        <div class="section">
            <h2>Visualizations</h2>
"""
        
        for name, fig in figures.items():
            if isinstance(fig, dict) and 'bytes' in fig:
                # Convert bytes to base64 for embedding
                img_base64 = base64.b64encode(fig['bytes']).decode('utf-8')
                
                html += f"""            <div class="figure">
                <h3>{name.replace('_', ' ').title()}</h3>
                <img src="data:image/png;base64,{img_base64}" alt="{name}">
            </div>
"""
        
        html += """        </div>
"""
    
    # Add results details
    if results:
        for category, result in results.items():
            if result:
                html += f"""        <div class="section">
            <h2>{category.replace('_', ' ').title()} Analysis</h2>
            <pre>{json.dumps(result, indent=4)}</pre>
        </div>
"""
    
    # Close HTML
    html += """    </div>
</body>
</html>
"""
    
    return html

def _generate_basic_info_section(basic_results, data=None) -> str:
    """Generate HTML section for basic dataset information."""
    # Basic implementation - in a real implementation this would be much more detailed
    html = "<h2>Basic Dataset Information</h2>\n"
    
    if basic_results.get('shape'):
        rows = basic_results['shape'].get('rows', 0)
        cols = basic_results['shape'].get('columns', 0)
        html += f"<p>Dataset dimensions: {rows} rows × {cols} columns</p>\n"
    
    return html

def _generate_missing_values_section(missing_results) -> str:
    """Generate HTML section for missing values analysis."""
    # Basic implementation
    html = "<h2>Missing Values Analysis</h2>\n"
    
    if missing_results.get('overall'):
        missing_pct = missing_results['overall'].get('missing_percentage', 0)
        html += f"<p>Overall missing: {missing_pct:.2f}% of all values</p>\n"
    
    return html

def _generate_outliers_section(outlier_results) -> str:
    """Generate HTML section for outlier analysis."""
    # Basic implementation
    html = "<h2>Outlier Analysis</h2>\n"
    
    if outlier_results.get('summary'):
        total = outlier_results['summary'].get('total_outliers', 0)
        cols = outlier_results['summary'].get('columns_with_outliers', 0)
        html += f"<p>Detected {total} outliers across {cols} columns</p>\n"
    
    return html

def _generate_correlation_section(correlation_results) -> str:
    """Generate HTML section for correlation analysis."""
    # Basic implementation
    html = "<h2>Correlation Analysis</h2>\n"
    
    if correlation_results.get('strong_correlations'):
        count = len(correlation_results['strong_correlations'])
        html += f"<p>Found {count} strong correlations between features</p>\n"
    
    return html

def _generate_distribution_section(distribution_results) -> str:
    """Generate HTML section for distribution analysis."""
    # Basic implementation
    html = "<h2>Distribution Analysis</h2>\n"
    
    if distribution_results.get('summary'):
        numeric = distribution_results['summary'].get('numeric_columns_analyzed', 0)
        categorical = distribution_results['summary'].get('categorical_columns_analyzed', 0)
        html += f"<p>Analyzed distributions of {numeric} numeric and {categorical} categorical columns</p>\n"
    
    return html

def _generate_target_analysis_section(target_results, target_variable) -> str:
    """Generate HTML section for target variable analysis."""
    # Basic implementation
    html = f"<h2>Target Analysis: {target_variable}</h2>\n"
    
    if target_results.get('target_type'):
        target_type = target_results['target_type']
        html += f"<p>Target variable type: {target_type}</p>\n"
    
    if target_results.get('top_features'):
        count = len(target_results['top_features'])
        html += f"<p>Identified {count} most important features for the target</p>\n"
    
    return html

def _embed_figures(figures) -> str:
    """Embed figures into HTML."""
    if not figures:
        return ""
    
    html = "<h2>Visualizations</h2>\n"
    
    for name, fig in figures.items():
        if isinstance(fig, dict) and 'bytes' in fig:
            # Convert bytes to base64 for embedding
            img_base64 = base64.b64encode(fig['bytes']).decode('utf-8')
            
            html += f"""<div class="figure">
    <h3>{name.replace('_', ' ').title()}</h3>
    <img src="data:image/png;base64,{img_base64}" alt="{name}">
</div>
"""
    
    return html

def _generate_code_snippets(results, target_variable) -> Dict[str, str]:
    """Generate Python code snippets for replicating the analysis."""
    code_snippets = {}
    
    # Basic snippet for loading and analyzing data
    code_snippets['basic'] = """<div class="code-section">
<h3>Basic Analysis Code</h3>
<pre class="code">
import pandas as pd
from eda_automator.core import EDACore

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize EDA Automator
eda = EDACore(df)

# Run basic analysis
basic_results = eda.run_basic_analysis()
</pre>
</div>"""
    
    # Add more specific snippets if results are available
    
    return code_snippets 