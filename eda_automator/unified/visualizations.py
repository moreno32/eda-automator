"""
Visualization functions for unified EDA reports.

This module contains functions for creating various visualizations
for exploratory data analysis reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import io
import base64

from .config import (
    MAX_CATEGORIES_FOR_BAR,
    MAX_CATEGORIES_FOR_PIE,
    MAX_CATEGORIES_FOR_HEATMAP,
    FIGURE_SIZE,
    FIGURE_DPI
)

# Set visualization style
def set_visualization_style():
    """Set consistent style for visualizations."""
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

# Initialize the style
set_visualization_style()

def create_data_overview_visualizations(df, column_types):
    """
    Create overview visualizations for the dataset.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        column_types (dict): Dictionary with column types
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    visualizations = {}
    
    # Data type distribution pie chart
    visualizations['data_types'] = create_data_types_pie(df, column_types)
    
    # Missing data heatmap
    visualizations['missing_data'] = create_missing_data_heatmap(df)
    
    # Data completeness bar chart
    visualizations['data_completeness'] = create_data_completeness_chart(df)
    
    return visualizations

def create_data_types_pie(df, column_types):
    """Create a pie chart showing the distribution of column data types."""
    # Count columns by type
    type_counts = {
        'Numeric': len(column_types.get('numeric', [])),
        'Categorical': len(column_types.get('categorical', [])),
        'DateTime': len(column_types.get('datetime', [])),
        'Text': len(column_types.get('text', [])),
        'Boolean': len(column_types.get('boolean', [])),
        'ID': len(column_types.get('id', []))
    }
    
    # Remove types with zero counts
    type_counts = {k: v for k, v in type_counts.items() if v > 0}
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.pie(
        type_counts.values(),
        labels=type_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        wedgeprops={'edgecolor': 'white'}
    )
    plt.title('Column Data Types Distribution')
    plt.tight_layout()
    
    # Convert to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=FIGURE_DPI)
    plt.close()
    
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return img_base64

def create_missing_data_heatmap(df):
    """Create a heatmap showing missing data patterns."""
    # Only include columns with missing values
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
    
    if not cols_with_missing:
        # No missing data, create empty plot
        plt.figure(figsize=(10, 2))
        plt.text(0.5, 0.5, 'No Missing Data', 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
    else:
        # Limit the number of columns to display
        if len(cols_with_missing) > MAX_CATEGORIES_FOR_HEATMAP:
            # Sort columns by missing percentage
            missing_pcts = df[cols_with_missing].isnull().mean().sort_values(ascending=False)
            cols_with_missing = list(missing_pcts.index[:MAX_CATEGORIES_FOR_HEATMAP])
        
        # Create a mask for missing values
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            df[cols_with_missing].isnull(),
            cmap=['#EEEEEE', '#FF5555'],
            cbar=False,
            yticklabels=False
        )
        plt.title('Missing Data Heatmap')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.tight_layout()
    
    # Convert to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=FIGURE_DPI)
    plt.close()
    
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return img_base64

def create_data_completeness_chart(df):
    """Create a bar chart showing data completeness by column."""
    # Calculate completeness percentage
    completeness = (1 - df.isnull().mean()) * 100
    
    # Sort by completeness
    completeness = completeness.sort_values()
    
    # Limit number of columns to display
    if len(completeness) > MAX_CATEGORIES_FOR_BAR:
        # Keep only the columns with least completeness
        completeness = completeness[:MAX_CATEGORIES_FOR_BAR]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Add color gradient
    bars = plt.barh(completeness.index, completeness.values)
    
    # Add color based on completeness
    for i, bar in enumerate(bars):
        if completeness.values[i] >= 95:
            bar.set_color('#28a745')  # Green for high completeness
        elif completeness.values[i] >= 80:
            bar.set_color('#17a2b8')  # Blue for medium completeness
        elif completeness.values[i] >= 50:
            bar.set_color('#ffc107')  # Yellow for low completeness
        else:
            bar.set_color('#dc3545')  # Red for very low completeness
    
    # Add percentage labels
    for i, v in enumerate(completeness.values):
        plt.text(v + 1, i, f'{v:.1f}%', va='center')
    
    plt.title('Data Completeness by Column')
    plt.xlabel('Completeness (%)')
    plt.xlim(0, 105)  # Leave space for percentage labels
    plt.tight_layout()
    
    # Convert to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=FIGURE_DPI)
    plt.close()
    
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return img_base64

def create_categorical_visualizations(df, categorical_cols, target=None):
    """
    Create visualizations for categorical variables.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        categorical_cols (list): List of categorical column names
        target (str): Target variable name (optional)
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    visualizations = {}
    
    # Limit to first N categorical columns to avoid too many plots
    if len(categorical_cols) > MAX_CATEGORIES_FOR_BAR:
        categorical_cols = categorical_cols[:MAX_CATEGORIES_FOR_BAR]
    
    # Create individual plots for each categorical column
    for col in categorical_cols:
        # Skip if column has too many unique values
        if df[col].nunique() > MAX_CATEGORIES_FOR_BAR:
            continue
            
        # Frequency plot
        visualizations[f'{col}_frequency'] = create_categorical_frequency_plot(df, col)
        
        # Target relationship plot if target is provided
        if target is not None and target in df.columns:
            visualizations[f'{col}_{target}'] = create_categorical_target_plot(df, col, target)
    
    return visualizations

def create_categorical_frequency_plot(df, column):
    """Create a bar plot showing the frequency of each category."""
    # Get value counts
    value_counts = df[column].value_counts().sort_values(ascending=False)
    
    # Limit number of categories to display
    if len(value_counts) > MAX_CATEGORIES_FOR_BAR:
        other_count = value_counts[MAX_CATEGORIES_FOR_BAR:].sum()
        value_counts = value_counts[:MAX_CATEGORIES_FOR_BAR]
        value_counts['Other'] = other_count
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(value_counts.index.astype(str), value_counts.values)
    
    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:,}',
                ha='center', va='bottom', rotation=0)
    
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Convert to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=FIGURE_DPI)
    plt.close()
    
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return img_base64

def create_categorical_target_plot(df, column, target):
    """Create a plot showing the relationship between a categorical column and the target."""
    plt.figure(figsize=(10, 6))
    
    # Check if target is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[target]):
        # For numeric target, use box plot
        sns.boxplot(x=column, y=target, data=df)
        plt.title(f'Distribution of {target} by {column}')
    else:
        # For categorical target, use stacked bar chart
        # Calculate percentages
        cross_tab = pd.crosstab(df[column], df[target], normalize='index') * 100
        
        # Plot stacked bar chart
        cross_tab.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title(f'Distribution of {target} by {column}')
        plt.ylabel('Percentage (%)')
        plt.legend(title=target)
    
    plt.xlabel(column)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Convert to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=FIGURE_DPI)
    plt.close()
    
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return img_base64

def create_numerical_visualizations(df, numerical_cols, target=None):
    """
    Create visualizations for numerical variables.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        numerical_cols (list): List of numerical column names
        target (str): Target variable name (optional)
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    visualizations = {}
    
    # Limit to first N numerical columns to avoid too many plots
    if len(numerical_cols) > MAX_CATEGORIES_FOR_BAR:
        numerical_cols = numerical_cols[:MAX_CATEGORIES_FOR_BAR]
    
    # Create individual plots for each numerical column
    for col in numerical_cols:
        # Distribution plot
        visualizations[f'{col}_distribution'] = create_numerical_distribution_plot(df, col)
        
        # Target relationship plot if target is provided
        if target is not None and target in df.columns and target != col:
            visualizations[f'{col}_{target}'] = create_numerical_target_plot(df, col, target)
    
    return visualizations

def create_numerical_distribution_plot(df, column):
    """Create a distribution plot for a numerical column."""
    plt.figure(figsize=(10, 6))
    
    # Check if values are mostly unique (potential ID column)
    if df[column].nunique() / len(df) > 0.9:
        # For ID-like columns, just show range and count
        plt.text(0.5, 0.5, f'Column appears to be an ID or unique identifier.\n'
                           f'Range: {df[column].min()} - {df[column].max()}\n'
                           f'Unique values: {df[column].nunique()}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=12,
                 transform=plt.gca().transAxes)
        plt.axis('off')
    else:
        # For regular numeric columns, show histogram and KDE
        sns.histplot(df[column].dropna(), kde=True)
        
        # Add vertical lines for quartiles
        plt.axvline(df[column].median(), color='red', linestyle='--', label='Median')
        plt.axvline(df[column].quantile(0.25), color='green', linestyle=':', label='Q1/Q3')
        plt.axvline(df[column].quantile(0.75), color='green', linestyle=':')
        
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend()
    
    plt.tight_layout()
    
    # Convert to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=FIGURE_DPI)
    plt.close()
    
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return img_base64

def create_numerical_target_plot(df, column, target):
    """Create a plot showing the relationship between a numerical column and the target."""
    plt.figure(figsize=(10, 6))
    
    # Check if target is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[target]):
        # For numeric target, use scatter plot with regression line
        sns.regplot(x=column, y=target, data=df, scatter_kws={'alpha': 0.3})
        plt.title(f'Relationship between {column} and {target}')
    else:
        # For categorical target, use violin plot
        sns.violinplot(x=target, y=column, data=df)
        plt.title(f'Distribution of {column} by {target}')
    
    plt.tight_layout()
    
    # Convert to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=FIGURE_DPI)
    plt.close()
    
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return img_base64

def create_correlation_heatmap(df, numeric_cols=None):
    """
    Create a correlation heatmap for numerical variables.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        numeric_cols (list): List of numerical column names (optional)
    
    Returns:
        str: Base64 encoded image
    """
    # Use provided numeric columns or find them
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # If there are too many columns, limit to the ones that have
    # the strongest correlations with others
    if len(numeric_cols) > MAX_CATEGORIES_FOR_HEATMAP:
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # For each column, get the average correlation with other columns
        avg_corr = corr_matrix.mean()
        
        # Sort columns by average correlation and keep top N
        numeric_cols = avg_corr.sort_values(ascending=False).index[:MAX_CATEGORIES_FOR_HEATMAP]
    
    # Calculate correlation matrix
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create custom colormap (blue to white to red)
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Correlation Heatmap')
        plt.tight_layout()
    else:
        # Not enough numeric columns for correlation
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Not enough numeric columns for correlation analysis',
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14)
        plt.axis('off')
    
    # Convert to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=FIGURE_DPI)
    plt.close()
    
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return img_base64

# Funciones específicas para integración con unified_eda_modular.py

def generate_basic_visualizations(df, eda):
    """
    Generate basic visualizations for the EDA report.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        eda (EDAAutomator): EDA instance with analysis results
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    plots = {}
    
    # Data overview visualizations
    plots['overview'] = create_data_overview_visualizations(df, eda.column_types)
    
    # Correlation heatmap for numeric columns
    if eda.column_types.get('numeric', []):
        plots['correlation'] = create_correlation_heatmap(df, eda.column_types['numeric'])
    
    return plots

def generate_univariate_plots(df, eda):
    """
    Generate univariate plots for all variables.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        eda (EDAAutomator): EDA instance with analysis results
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    plots = {}
    
    # Categorical visualizations
    if eda.column_types.get('categorical', []):
        plots['categorical'] = create_categorical_visualizations(
            df, eda.column_types['categorical']
        )
    
    # Numeric visualizations
    if eda.column_types.get('numeric', []):
        plots['numeric'] = create_numerical_visualizations(
            df, eda.column_types['numeric']
        )
    
    return plots

def generate_bivariate_plots(df, target_variable, eda):
    """
    Generate bivariate plots for relationships with target variable.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        target_variable (str): Name of the target variable
        eda (EDAAutomator): EDA instance with analysis results
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    plots = {}
    
    # Only proceed if target variable exists in the data
    if target_variable not in df.columns:
        return plots
    
    # Categorical variables vs target
    if eda.column_types.get('categorical', []):
        categorical_cols = [col for col in eda.column_types['categorical'] if col != target_variable]
        if categorical_cols:
            plots['categorical_target'] = create_categorical_visualizations(
                df, categorical_cols, target_variable
            )
    
    # Numeric variables vs target
    if eda.column_types.get('numeric', []):
        numeric_cols = [col for col in eda.column_types['numeric'] if col != target_variable]
        if numeric_cols:
            plots['numeric_target'] = create_numerical_visualizations(
                df, numeric_cols, target_variable
            )
    
    return plots 