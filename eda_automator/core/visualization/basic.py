"""
Basic visualization functions for EDA Automator

This module provides functions for visualizing basic dataset information.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from eda_automator.core.utils import get_logger
from .figures import set_plot_style

# Initialize logger
logger = get_logger()

def plot_basic_info(
    df: pd.DataFrame,
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate basic information plots for the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data
    results : dict
        Results from basic analysis
        
    Returns
    -------
    dict
        Dictionary of figure objects
    """
    logger.info("Generating basic information plots")
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.error("matplotlib is required for visualization")
        return {"error": "matplotlib is required for visualization"}
    
    # Set plot style
    set_plot_style()
    
    # Initialize figures dictionary
    figures = {}
    
    # 1. Data types pie chart
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        column_types = results.get('column_types', {})
        
        # Count each type
        type_counts = {
            'Numeric': len(column_types.get('numeric', [])),
            'Categorical': len(column_types.get('categorical', [])),
            'Datetime': len(column_types.get('datetime', [])),
            'Boolean': len(column_types.get('boolean', []))
        }
        
        # Remove types with zero count
        type_counts = {k: v for k, v in type_counts.items() if v > 0}
        
        if type_counts:
            # Create pie chart
            explode = [0.05] * len(type_counts)  # Small explode for all slices
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(type_counts)]
            
            # Draw pie chart
            wedges, texts, autotexts = ax.pie(
                type_counts.values(),
                labels=type_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                colors=colors,
                shadow=False
            )
            
            # Make text easier to read
            for text in texts:
                text.set_fontsize(12)
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_color('white')
            
            # Equal aspect ratio ensures the pie chart is circular
            ax.axis('equal')
            
            # Add title
            plt.title('Distribution of Column Types', pad=20)
            
            # Add column count to title
            total_columns = sum(type_counts.values())
            fig.text(0.5, 0.01, f'Total columns: {total_columns}', ha='center')
            
            # Store figure
            figures['data_types_pie'] = fig
        else:
            logger.warning("No data type information available for plotting")
    except Exception as e:
        logger.error(f"Error generating data types pie chart: {str(e)}")
    
    # 2. Missing values overview
    try:
        missing_counts = results.get('missing_counts', {})
        missing_percentages = results.get('missing_percentages', {})
        
        if missing_counts and any(missing_counts.values()):
            # Sort by missing percentage (descending)
            missing_data = pd.DataFrame({
                'count': pd.Series(missing_counts),
                'percentage': pd.Series(missing_percentages)
            }).sort_values(by='percentage', ascending=False)
            
            # Select top 15 columns with missing values
            missing_data = missing_data[missing_data['count'] > 0].head(15)
            
            if not missing_data.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bars = ax.barh(
                    missing_data.index,
                    missing_data['percentage'],
                    color='#ff7f0e',
                    alpha=0.8
                )
                
                # Add value labels
                for bar, count, pct in zip(bars, missing_data['count'], missing_data['percentage']):
                    ax.text(
                        min(pct + 2, 100),  # Position
                        bar.get_y() + bar.get_height()/2,  # Y position (middle of bar)
                        f"{count} ({pct:.1f}%)",  # Text
                        va='center',
                        color='black' if pct < 70 else 'white',
                        fontweight='bold'
                    )
                
                # Set labels and title
                ax.set_xlabel('Missing Values (%)')
                ax.set_ylabel('Column')
                ax.set_title('Top Columns with Missing Values')
                
                # Set x-axis limit
                ax.set_xlim(0, 100)
                
                # Add grid
                ax.xaxis.grid(True, alpha=0.3)
                
                # Remove frame and tick marks on y-axis
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', length=0)
                
                # Adjust layout
                plt.tight_layout()
                
                # Store figure
                figures['missing_values_overview'] = fig
        else:
            logger.info("No missing values in the dataset")
    except Exception as e:
        logger.error(f"Error generating missing values overview: {str(e)}")
    
    # 3. Data preview table
    try:
        preview_data = results.get('preview', [])
        
        if preview_data:
            # Convert preview to DataFrame
            preview_df = pd.DataFrame(preview_data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, len(preview_df) * 0.5 + 1))
            
            # Hide axes
            ax.axis('off')
            
            # Create table
            table = ax.table(
                cellText=preview_df.values,
                colLabels=preview_df.columns,
                loc='center',
                cellLoc='center'
            )
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style header
            for k, cell in table.get_celld().items():
                if k[0] == 0:  # Header row
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#1f77b4')
                elif k[0] % 2 == 0:  # Even rows
                    cell.set_facecolor('#f5f5f5')
            
            # Add title
            plt.title('Data Preview (First 5 rows)', pad=20)
            
            # Adjust layout
            plt.tight_layout()
            
            # Store figure
            figures['data_preview'] = fig
    except Exception as e:
        logger.error(f"Error generating data preview table: {str(e)}")
    
    # 4. Memory usage by data type
    try:
        memory_usage = df.memory_usage(deep=True)
        memory_by_column = memory_usage.drop('Index')  # Remove Index
        
        if not memory_by_column.empty:
            # Get dtypes
            dtypes = df.dtypes.astype(str)
            
            # Group by simplified data type
            memory_by_type = {}
            for col, mem in memory_by_column.items():
                dtype = dtypes[col]
                if 'int' in dtype:
                    dtype_group = 'Integer'
                elif 'float' in dtype:
                    dtype_group = 'Float'
                elif 'object' in dtype:
                    dtype_group = 'Object'
                elif 'datetime' in dtype:
                    dtype_group = 'Datetime'
                elif 'bool' in dtype:
                    dtype_group = 'Boolean'
                elif 'category' in dtype:
                    dtype_group = 'Category'
                else:
                    dtype_group = 'Other'
                
                if dtype_group in memory_by_type:
                    memory_by_type[dtype_group] += mem
                else:
                    memory_by_type[dtype_group] = mem
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Convert to MB for readability
            memory_by_type_mb = {k: v / (1024 * 1024) for k, v in memory_by_type.items()}
            
            # Sort by memory usage
            memory_by_type_mb = dict(sorted(memory_by_type_mb.items(), key=lambda x: x[1], reverse=True))
            
            # Colors
            colors = {
                'Integer': '#1f77b4',
                'Float': '#ff7f0e',
                'Object': '#2ca02c',
                'Datetime': '#d62728',
                'Boolean': '#9467bd',
                'Category': '#8c564b',
                'Other': '#7f7f7f'
            }
            
            # Create bar chart
            bars = ax.bar(
                memory_by_type_mb.keys(),
                memory_by_type_mb.values(),
                color=[colors.get(dtype, '#7f7f7f') for dtype in memory_by_type_mb.keys()]
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.1,
                    f"{height:.2f} MB",
                    ha='center',
                    va='bottom',
                    rotation=0
                )
            
            # Set labels and title
            ax.set_xlabel('Data Type')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage by Data Type')
            
            # Add total memory usage
            total_memory_mb = sum(memory_by_type_mb.values())
            fig.text(0.5, 0.01, f'Total Memory: {total_memory_mb:.2f} MB', ha='center')
            
            # Adjust layout
            plt.tight_layout()
            
            # Store figure
            figures['memory_usage'] = fig
    except Exception as e:
        logger.error(f"Error generating memory usage chart: {str(e)}")
    
    logger.info(f"Generated {len(figures)} basic information plots")
    return figures 