"""
Markdown report generation module for EDA Automator

This module provides functions for generating Markdown reports from analysis results.
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

from eda_automator.core.utils import get_logger, format_number, format_percent, format_list

# Initialize logger
logger = get_logger()

def generate_markdown_report(
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
    Generate a Markdown report from the EDA results.
    
    Parameters
    ----------
    output_path : str
        Path to save the Markdown report
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
        Path to a custom Markdown template file
    include_code : bool, default True
        Whether to include Python code snippets in the report
        
    Returns
    -------
    str
        Path to the generated Markdown report
    """
    logger.info(f"Generating Markdown report to {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Get default template if provided
    if template_path is not None:
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
                
            # Generate report content based on template
            # TODO: Implement template-based report generation
            md_content = _generate_basic_markdown(data, results, figures, target_variable, settings, include_code)
        except Exception as e:
            logger.error(f"Error loading template: {str(e)}")
            logger.warning("Using basic template")
            md_content = _generate_basic_markdown(data, results, figures, target_variable, settings, include_code)
    else:
        # Generate basic markdown report
        md_content = _generate_basic_markdown(data, results, figures, target_variable, settings, include_code)
    
    # Write markdown to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    logger.info(f"Generated Markdown report at {output_path}")
    return output_path

def _generate_basic_markdown(
    data=None,
    results=None,
    figures=None,
    target_variable=None,
    settings=None,
    include_code=True
) -> str:
    """
    Generate a basic Markdown report.
    
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
    include_code : bool, default True
        Whether to include Python code snippets in the report
        
    Returns
    -------
    str
        Markdown content
    """
    # Create markdown content
    md = "# EDA Automator Report\n\n"
    
    # Add metadata
    md += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add data info
    if data is not None:
        md += f"**Dataset:** {data.shape[0]} rows × {data.shape[1]} columns\n\n"
    
    # Add target variable info
    if target_variable:
        md += f"**Target variable:** {target_variable}\n\n"
    
    # Add table of contents
    md += "## Table of Contents\n\n"
    md += "1. [Analysis Summary](#analysis-summary)\n"
    
    if results and 'basic' in results:
        md += "2. [Basic Dataset Information](#basic-dataset-information)\n"
    
    if results and 'missing_values' in results:
        md += "3. [Missing Values Analysis](#missing-values-analysis)\n"
    
    if results and 'outliers' in results:
        md += "4. [Outlier Analysis](#outlier-analysis)\n"
    
    if results and 'correlation' in results:
        md += "5. [Correlation Analysis](#correlation-analysis)\n"
    
    if results and 'distribution' in results:
        md += "6. [Distribution Analysis](#distribution-analysis)\n"
    
    if results and 'target_analysis' in results:
        md += "7. [Target Analysis](#target-analysis)\n"
    
    if figures:
        md += "8. [Visualizations](#visualizations)\n"
    
    if include_code:
        md += "9. [Code Snippets](#code-snippets)\n"
    
    md += "\n"
    
    # Add analysis summary
    md += "## Analysis Summary\n\n"
    
    if results:
        md += "The following analyses were performed:\n\n"
        for category, result in results.items():
            if result:
                md += f"- ✓ {category.replace('_', ' ').title()}\n"
    else:
        md += "No analysis results available.\n"
    
    md += "\n"
    
    # Add basic info section
    if results and 'basic' in results:
        md += _generate_basic_info_markdown(results['basic'], data)
    
    # Add missing values section
    if results and 'missing_values' in results:
        md += _generate_missing_values_markdown(results['missing_values'])
    
    # Add outliers section
    if results and 'outliers' in results:
        md += _generate_outliers_markdown(results['outliers'])
    
    # Add correlation section
    if results and 'correlation' in results:
        md += _generate_correlation_markdown(results['correlation'])
    
    # Add distribution section
    if results and 'distribution' in results:
        md += _generate_distribution_markdown(results['distribution'])
    
    # Add target analysis section
    if results and 'target_analysis' in results and target_variable:
        md += _generate_target_analysis_markdown(results['target_analysis'], target_variable)
    
    # Add figures
    if figures:
        md += _embed_figures_markdown(figures)
    
    # Add code snippets
    if include_code:
        md += _generate_code_snippets_markdown(results, target_variable)
    
    return md

def _generate_basic_info_markdown(basic_results, data=None) -> str:
    """Generate Markdown section for basic dataset information."""
    md = "## Basic Dataset Information\n\n"
    
    if basic_results.get('shape'):
        rows = basic_results['shape'].get('rows', 0)
        cols = basic_results['shape'].get('columns', 0)
        md += f"- **Dimensions:** {rows} rows × {cols} columns\n"
    
    if basic_results.get('dtypes'):
        md += "- **Data Types:**\n"
        for dtype, count in basic_results['dtypes'].items():
            md += f"  - {dtype}: {count} columns\n"
    
    if basic_results.get('memory_usage'):
        memory = basic_results['memory_usage'].get('total', 0)
        md += f"- **Memory Usage:** {format_number(memory, decimals=2)} bytes\n"
    
    if data is not None and basic_results.get('sample'):
        md += "\n### Sample Data\n\n"
        
        # Format sample as markdown table
        columns = data.columns.tolist()
        md += "| " + " | ".join(str(col) for col in columns) + " |\n"
        md += "| " + " | ".join(["---"] * len(columns)) + " |\n"
        
        # Add sample rows
        sample_size = min(5, data.shape[0])
        for i in range(sample_size):
            md += "| " + " | ".join(str(data.iloc[i][col]) for col in columns) + " |\n"
    
    md += "\n"
    return md

def _generate_missing_values_markdown(missing_results) -> str:
    """Generate Markdown section for missing values analysis."""
    md = "## Missing Values Analysis\n\n"
    
    if missing_results.get('overall'):
        missing_pct = missing_results['overall'].get('missing_percentage', 0)
        missing_count = missing_results['overall'].get('missing_count', 0)
        total = missing_results['overall'].get('total_values', 0)
        
        md += f"- **Overall Missing:** {format_percent(missing_pct)} ({missing_count:,} out of {total:,} values)\n\n"
    
    if missing_results.get('by_column'):
        md += "### Missing Values by Column\n\n"
        
        # Create a markdown table
        md += "| Column | Missing Count | Missing Percent |\n"
        md += "| ------ | ------------: | --------------: |\n"
        
        # Sort columns by missing percentage in descending order
        columns = sorted(
            missing_results['by_column'].items(),
            key=lambda x: x[1].get('missing_percentage', 0),
            reverse=True
        )
        
        # Add data for each column with missing values
        for column, stats in columns:
            if stats.get('missing_count', 0) > 0:
                missing_count = stats.get('missing_count', 0)
                missing_pct = stats.get('missing_percentage', 0)
                md += f"| {column} | {missing_count:,} | {format_percent(missing_pct)} |\n"
    
    md += "\n"
    return md

def _generate_outliers_markdown(outlier_results) -> str:
    """Generate Markdown section for outlier analysis."""
    md = "## Outlier Analysis\n\n"
    
    if outlier_results.get('summary'):
        total = outlier_results['summary'].get('total_outliers', 0)
        cols = outlier_results['summary'].get('columns_with_outliers', 0)
        method = outlier_results['summary'].get('method', 'Unknown')
        
        md += f"- **Detection Method:** {method}\n"
        md += f"- **Total Outliers:** {total:,} outliers across {cols} columns\n\n"
    
    if outlier_results.get('by_column'):
        md += "### Outliers by Column\n\n"
        
        # Create a markdown table
        md += "| Column | Outlier Count | Outlier Percent | Lower Bound | Upper Bound |\n"
        md += "| ------ | ------------: | --------------: | ----------: | ----------: |\n"
        
        # Sort columns by outlier count in descending order
        columns = sorted(
            outlier_results['by_column'].items(),
            key=lambda x: x[1].get('outlier_count', 0),
            reverse=True
        )
        
        # Add data for each column with outliers
        for column, stats in columns:
            if stats.get('outlier_count', 0) > 0:
                outlier_count = stats.get('outlier_count', 0)
                outlier_pct = stats.get('outlier_percentage', 0)
                lower_bound = stats.get('lower_bound', 'N/A')
                upper_bound = stats.get('upper_bound', 'N/A')
                
                if lower_bound != 'N/A':
                    lower_bound = format_number(lower_bound)
                
                if upper_bound != 'N/A':
                    upper_bound = format_number(upper_bound)
                
                md += f"| {column} | {outlier_count:,} | {format_percent(outlier_pct)} | {lower_bound} | {upper_bound} |\n"
    
    md += "\n"
    return md

def _generate_correlation_markdown(correlation_results) -> str:
    """Generate Markdown section for correlation analysis."""
    md = "## Correlation Analysis\n\n"
    
    if correlation_results.get('method'):
        md += f"- **Correlation Method:** {correlation_results['method']}\n\n"
    
    if correlation_results.get('strong_correlations'):
        md += "### Strong Feature Correlations\n\n"
        
        # Create a markdown table
        md += "| Feature 1 | Feature 2 | Correlation |\n"
        md += "| --------- | --------- | ----------: |\n"
        
        # Sort correlations by absolute value in descending order
        correlations = sorted(
            correlation_results['strong_correlations'],
            key=lambda x: abs(x.get('correlation', 0)),
            reverse=True
        )
        
        # Add data for each correlation pair
        for corr in correlations:
            feature1 = corr.get('feature1', 'Unknown')
            feature2 = corr.get('feature2', 'Unknown')
            correlation = corr.get('correlation', 0)
            
            md += f"| {feature1} | {feature2} | {format_number(correlation, decimals=3)} |\n"
    
    md += "\n"
    return md

def _generate_distribution_markdown(distribution_results) -> str:
    """Generate Markdown section for distribution analysis."""
    md = "## Distribution Analysis\n\n"
    
    if distribution_results.get('summary'):
        numeric = distribution_results['summary'].get('numeric_columns_analyzed', 0)
        categorical = distribution_results['summary'].get('categorical_columns_analyzed', 0)
        
        md += f"- Analyzed distributions of {numeric} numeric and {categorical} categorical columns\n\n"
    
    if distribution_results.get('numeric'):
        md += "### Numeric Columns Statistics\n\n"
        
        # Create a markdown table
        md += "| Column | Mean | Median | Std Dev | Min | Max | Skew | Kurtosis |\n"
        md += "| ------ | ---: | -----: | ------: | --: | --: | ---: | -------: |\n"
        
        # Add data for each numeric column
        for column, stats in distribution_results['numeric'].items():
            mean = format_number(stats.get('mean', 0))
            median = format_number(stats.get('median', 0))
            std = format_number(stats.get('std', 0))
            min_val = format_number(stats.get('min', 0))
            max_val = format_number(stats.get('max', 0))
            skew = format_number(stats.get('skew', 0), decimals=3)
            kurtosis = format_number(stats.get('kurtosis', 0), decimals=3)
            
            md += f"| {column} | {mean} | {median} | {std} | {min_val} | {max_val} | {skew} | {kurtosis} |\n"
        
        md += "\n"
    
    if distribution_results.get('categorical'):
        md += "### Categorical Columns Statistics\n\n"
        
        # Process each categorical column
        for column, stats in distribution_results['categorical'].items():
            md += f"#### {column}\n\n"
            
            if stats.get('unique_values') is not None:
                unique = stats.get('unique_values', 0)
                md += f"- **Unique Values:** {unique}\n"
            
            if stats.get('top_categories'):
                md += "- **Top Categories:**\n\n"
                
                # Create a markdown table
                md += "| Value | Count | Percentage |\n"
                md += "| ----- | ----: | ---------: |\n"
                
                # Add data for top categories
                for category in stats['top_categories']:
                    value = category.get('value', 'Unknown')
                    count = category.get('count', 0)
                    percentage = category.get('percentage', 0)
                    
                    md += f"| {value} | {count:,} | {format_percent(percentage)} |\n"
                
                md += "\n"
    
    md += "\n"
    return md

def _generate_target_analysis_markdown(target_results, target_variable) -> str:
    """Generate Markdown section for target variable analysis."""
    md = f"## Target Analysis\n\n"
    md += f"Target Variable: **{target_variable}**\n\n"
    
    if target_results.get('target_type'):
        target_type = target_results['target_type']
        md += f"- **Target Type:** {target_type}\n\n"
    
    if target_results.get('distribution'):
        md += "### Target Distribution\n\n"
        
        if target_type == 'categorical':
            # Create a markdown table for categorical target
            md += "| Value | Count | Percentage |\n"
            md += "| ----- | ----: | ---------: |\n"
            
            # Add data for each category
            for category in target_results['distribution'].get('categories', []):
                value = category.get('value', 'Unknown')
                count = category.get('count', 0)
                percentage = category.get('percentage', 0)
                
                md += f"| {value} | {count:,} | {format_percent(percentage)} |\n"
        else:
            # For numeric target, show statistics
            stats = target_results['distribution'].get('statistics', {})
            
            md += f"- **Mean:** {format_number(stats.get('mean', 0))}\n"
            md += f"- **Median:** {format_number(stats.get('median', 0))}\n"
            md += f"- **Std Dev:** {format_number(stats.get('std', 0))}\n"
            md += f"- **Min:** {format_number(stats.get('min', 0))}\n"
            md += f"- **Max:** {format_number(stats.get('max', 0))}\n"
            md += f"- **Skew:** {format_number(stats.get('skew', 0), decimals=3)}\n"
            md += f"- **Kurtosis:** {format_number(stats.get('kurtosis', 0), decimals=3)}\n"
        
        md += "\n"
    
    if target_results.get('top_features'):
        md += "### Top Features for Target\n\n"
        
        # Create a markdown table
        md += "| Feature | Importance |\n"
        md += "| ------- | ---------: |\n"
        
        # Add data for each feature
        for feature in target_results['top_features']:
            name = feature.get('name', 'Unknown')
            importance = feature.get('importance', 0)
            
            md += f"| {name} | {format_number(importance, decimals=4)} |\n"
    
    md += "\n"
    return md

def _embed_figures_markdown(figures) -> str:
    """Embed figures into Markdown."""
    if not figures:
        return ""
    
    # Check if any figures need to be saved as separate files
    figures_dir = None
    
    md = "## Visualizations\n\n"
    
    # Process each figure
    for name, fig in figures.items():
        title = name.replace('_', ' ').title()
        md += f"### {title}\n\n"
        
        # For now, just note that figures are available
        md += f"*Figure: {title}*\n\n"
        
        # Future enhancement: implement figure embedding for markdown
        # This would require saving the figures as image files
    
    return md

def _generate_code_snippets_markdown(results, target_variable) -> str:
    """Generate Python code snippets for replicating the analysis."""
    md = "## Code Snippets\n\n"
    md += "The following code snippets can be used to reproduce the analysis:\n\n"
    
    # Basic snippet for loading and analyzing data
    md += "### Basic Analysis\n\n"
    md += "```python\n"
    md += "import pandas as pd\n"
    md += "from eda_automator.core import EDACore\n\n"
    md += "# Load your data\n"
    md += "df = pd.read_csv('your_data.csv')\n\n"
    md += "# Initialize EDA Automator\n"
    md += "eda = EDACore(df)\n\n"
    md += "# Run basic analysis\n"
    md += "basic_results = eda.run_basic_analysis()\n"
    md += "```\n\n"
    
    # Add more specific snippets if results are available
    if results and 'missing_values' in results:
        md += "### Missing Values Analysis\n\n"
        md += "```python\n"
        md += "# Run missing values analysis\n"
        md += "missing_results = eda.run_missing_analysis()\n"
        md += "```\n\n"
    
    if results and 'outliers' in results:
        md += "### Outlier Analysis\n\n"
        md += "```python\n"
        md += "# Run outlier analysis\n"
        md += "outlier_results = eda.run_outlier_analysis()\n"
        md += "```\n\n"
    
    if results and 'correlation' in results:
        md += "### Correlation Analysis\n\n"
        md += "```python\n"
        md += "# Run correlation analysis\n"
        md += "correlation_results = eda.run_correlation_analysis()\n"
        md += "```\n\n"
    
    if results and 'distribution' in results:
        md += "### Distribution Analysis\n\n"
        md += "```python\n"
        md += "# Run distribution analysis\n"
        md += "distribution_results = eda.run_distribution_analysis()\n"
        md += "```\n\n"
    
    if results and 'target_analysis' in results and target_variable:
        md += "### Target Analysis\n\n"
        md += "```python\n"
        md += "# Run target analysis\n"
        md += f"target_results = eda.run_target_analysis(target_variable='{target_variable}')\n"
        md += "```\n\n"
    
    # Complete report generation
    md += "### Generate Report\n\n"
    md += "```python\n"
    md += "# Generate full report\n"
    md += "eda.generate_report(\n"
    md += "    output_path='eda_report.html',\n"
    md += "    format='html'\n"
    md += ")\n"
    md += "```\n\n"
    
    return md 