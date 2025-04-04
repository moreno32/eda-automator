"""
Unified visualizations module for EDA Automator.

This module contains functions to generate visualizations for the EDA Automator package.
"""

import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
sns.set_context('notebook')

def fig_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to convert
    
    Returns:
        str: Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def generate_basic_visualizations(df, eda):
    """
    Generate basic visualizations about the dataset.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        eda (EDAAutomator): EDA instance with analysis results
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    plots = {}
    
    # Data types distribution pie chart
    plots['data_types'] = generate_data_types_chart(eda)
    
    # Missing data heatmap
    if hasattr(eda, 'missing_data'):
        missing_vars = eda.missing_data.get('variables_with_missing', [])
        if missing_vars:
            plots['missing_heatmap'] = generate_missing_heatmap(df[missing_vars])
    
    # Correlation heatmap
    if hasattr(eda, 'correlations'):
        corr_matrix = eda.correlations.get('correlation_matrix', {})
        if corr_matrix:
            plots['correlation_heatmap'] = generate_correlation_heatmap(df)
    
    return plots

def generate_data_types_chart(eda):
    """
    Generate a pie chart showing the distribution of data types.
    
    Args:
        eda (EDAAutomator): EDA instance with analysis results
    
    Returns:
        str: Base64 encoded image
    """
    if not hasattr(eda, 'column_counts'):
        return None
    
    # Get counts
    counts = eda.column_counts
    
    # Filter out zero counts
    labels = []
    values = []
    
    for dtype, count in counts.items():
        if count > 0:
            labels.append(dtype)
            values.append(count)
    
    if not values:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors
    colors = sns.color_palette('viridis', len(labels))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        shadow=False
    )
    
    # Customize text
    plt.setp(autotexts, size=10, weight='bold')
    
    # Add legend
    ax.legend(
        wedges, 
        [f'{label} ({value})' for label, value in zip(labels, values)],
        title='Data Types',
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')
    
    plt.title('Distribution of Column Data Types', fontsize=14)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_missing_heatmap(df_missing):
    """
    Generate a heatmap of missing values.
    
    Args:
        df_missing (pandas.DataFrame): DataFrame with columns containing missing values
    
    Returns:
        str: Base64 encoded image
    """
    if df_missing.empty:
        return None
    
    # Create a mask of missing values
    mask = df_missing.isna()
    
    # Calculate percentage of missing values for each column
    missing_pct = mask.mean().round(4) * 100
    
    # Sort columns by percentage of missing values
    cols = missing_pct.sort_values(ascending=False).index
    
    # Take only the top 20 columns to avoid too many columns
    if len(cols) > 20:
        cols = cols[:20]
    
    # If only a few rows, take all rows
    if df_missing.shape[0] <= 100:
        sample = df_missing[cols]
    else:
        # Otherwise, sample rows
        sample = df_missing[cols].sample(n=min(100, df_missing.shape[0]), random_state=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        sample.isna(), 
        cmap='viridis', 
        cbar=False,
        ax=ax,
        yticklabels=False
    )
    
    # Add column labels with percentages
    ax.set_xticklabels([f'{col} ({missing_pct[col]:.1f}%)' for col in cols], rotation=45, ha='right')
    
    plt.title('Missing Values Heatmap (sample of data)', fontsize=14)
    plt.xlabel('Columns (with % missing)', fontsize=12)
    plt.ylabel('Rows (sample)', fontsize=12)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_correlation_heatmap(df):
    """
    Generate a heatmap of correlations between numeric variables.
    
    Args:
        df (pandas.DataFrame): DataFrame to visualize
    
    Returns:
        str: Base64 encoded image
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    # Limit to 20 columns for readability
    if len(numeric_cols) > 20:
        # Keep the most varying columns
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.index[:20].tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={'shrink': .5},
        annot=True,
        fmt='.2f',
        ax=ax
    )
    
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    
    return fig_to_base64(fig)

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

def create_categorical_visualizations(df, categorical_cols):
    """
    Create visualizations for categorical variables.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        categorical_cols (list): List of categorical column names
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    visualizations = {}
    
    # Limit to 20 columns to avoid too many plots
    if len(categorical_cols) > 20:
        # Select based on cardinality (prioritize those with fewer categories)
        cardinality = {col: df[col].nunique() for col in categorical_cols}
        categorical_cols = sorted(categorical_cols, key=lambda x: cardinality[x])[:20]
    
    for col in categorical_cols:
        # Skip if column has too many categories
        if df[col].nunique() > 30:
            continue
        
        # Count values and sort
        value_counts = df[col].value_counts().sort_values(ascending=False)
        
        # Take top 15 categories if there are more
        if len(value_counts) > 15:
            value_counts = value_counts.head(15)
        
        # Calculate percentages
        total = value_counts.sum()
        percentages = (value_counts / total * 100).round(1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        bars = ax.bar(
            value_counts.index.astype(str), 
            value_counts, 
            color=sns.color_palette('viridis', len(value_counts))
        )
        
        # Add percentage labels
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{percentage}%',
                ha='center', 
                va='bottom',
                rotation=0,
                fontsize=9
            )
        
        # Format plot
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save to dict
        visualizations[col] = fig_to_base64(fig)
    
    return visualizations

def create_numerical_visualizations(df, numeric_cols):
    """
    Create visualizations for numerical variables.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        numeric_cols (list): List of numeric column names
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    visualizations = {}
    
    # Limit to 20 columns to avoid too many plots
    if len(numeric_cols) > 20:
        # Select based on variance (prioritize those with more variance)
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.index[:20].tolist()
    
    for col in numeric_cols:
        # Skip if all values are missing
        if df[col].isna().all():
            continue
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram with KDE
        sns.histplot(df[col].dropna(), kde=True, ax=ax1, color='skyblue')
        ax1.set_title(f'Distribution of {col}', fontsize=12)
        ax1.set_xlabel(col, fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        
        # Boxplot
        sns.boxplot(x=df[col].dropna(), ax=ax2, color='skyblue')
        ax2.set_title(f'Boxplot of {col}', fontsize=12)
        ax2.set_xlabel(col, fontsize=10)
        
        # Add stats
        stats_text = (
            f"Mean: {df[col].mean():.2f}\n"
            f"Median: {df[col].median():.2f}\n"
            f"Std Dev: {df[col].std():.2f}\n"
            f"Min: {df[col].min():.2f}\n"
            f"Max: {df[col].max():.2f}"
        )
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(
            0.05, 0.95, stats_text, 
            transform=ax1.transAxes, 
            fontsize=9,
            verticalalignment='top', 
            bbox=props
        )
        
        plt.tight_layout()
        
        # Save to dict
        visualizations[col] = fig_to_base64(fig)
    
    return visualizations

def generate_bivariate_plots(df, target_variable, eda):
    """
    Generate bivariate plots between features and the target variable.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        target_variable (str): Name of the target variable
        eda (EDAAutomator): EDA instance with analysis results
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    if target_variable not in df.columns:
        return {}
    
    visualizations = {}
    target_dtype = df[target_variable].dtype
    
    # If target is numeric
    if np.issubdtype(target_dtype, np.number):
        # Correlations with numeric features
        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                         if col != target_variable]
        
        if numeric_cols:
            visualizations['numeric_correlations'] = visualize_numeric_correlations(
                df, target_variable, numeric_cols
            )
        
        # Relationship with categorical features
        categorical_cols = eda.column_types.get('categorical', [])
        if categorical_cols and len(categorical_cols) > 0:
            visualizations['categorical_relationships'] = visualize_categorical_to_numeric(
                df, target_variable, categorical_cols[:10]  # Limit to 10
            )
    
    # If target is categorical
    elif pd.api.types.is_categorical_dtype(target_dtype) or pd.api.types.is_object_dtype(target_dtype):
        # Only if there aren't too many categories
        if df[target_variable].nunique() <= 10:
            # Relationship with numeric features
            numeric_cols = eda.column_types.get('numeric', [])
            if numeric_cols:
                visualizations['numeric_by_target'] = visualize_numeric_by_category(
                    df, target_variable, numeric_cols[:10]  # Limit to 10
                )
            
            # Relationship with categorical features
            categorical_cols = [col for col in eda.column_types.get('categorical', [])
                                if col != target_variable]
            if categorical_cols:
                visualizations['categorical_associations'] = visualize_categorical_associations(
                    df, target_variable, categorical_cols[:5]  # Limit to 5
                )
    
    return visualizations

def visualize_numeric_correlations(df, target_variable, numeric_cols):
    """
    Visualize correlations between numeric features and a numeric target.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        target_variable (str): Name of the target variable
        numeric_cols (list): List of numeric column names
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    correlations = {}
    
    # Calculate correlations
    corr_values = {}
    for col in numeric_cols:
        corr = df[[col, target_variable]].corr().iloc[0, 1]
        corr_values[col] = corr
    
    # Sort by absolute correlation and take top 10
    sorted_cols = sorted(corr_values.items(), key=lambda x: abs(x[1]), reverse=True)
    top_cols = [col for col, _ in sorted_cols[:10]]
    
    # Create scatterplots for top features
    for col in top_cols:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatterplot with regression line
        sns.regplot(
            x=col, 
            y=target_variable, 
            data=df, 
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'},
            ax=ax
        )
        
        # Add correlation coefficient
        corr = corr_values[col]
        corr_text = f'Correlation: {corr:.2f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(
            0.05, 0.95, corr_text, 
            transform=ax.transAxes, 
            fontsize=12,
            verticalalignment='top', 
            bbox=props
        )
        
        # Format plot
        plt.title(f'{col} vs {target_variable}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel(target_variable, fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save to dict
        correlations[col] = fig_to_base64(fig)
    
    return correlations

def visualize_categorical_to_numeric(df, target_variable, categorical_cols):
    """
    Visualize relationship between categorical features and numeric target.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        target_variable (str): Name of the target variable
        categorical_cols (list): List of categorical column names
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    visualizations = {}
    
    for col in categorical_cols:
        # Skip if too many categories
        if df[col].nunique() > 15:
            continue
        
        # Group by the categorical column and calculate mean, std of target
        grouped = df.groupby(col)[target_variable].agg(['mean', 'std', 'count'])
        grouped = grouped.sort_values('mean', ascending=False)
        
        # Skip if any group has too few samples
        if grouped['count'].min() < 5:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        bars = ax.bar(
            grouped.index.astype(str), 
            grouped['mean'], 
            yerr=grouped['std'],
            capsize=5,
            color=sns.color_palette('viridis', len(grouped))
        )
        
        # Add count labels
        for bar, count in zip(bars, grouped['count']):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'n={count}',
                ha='center', 
                va='bottom',
                rotation=0,
                fontsize=9
            )
        
        # Format plot
        plt.title(f'Mean {target_variable} by {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel(f'Mean {target_variable}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save to dict
        visualizations[col] = fig_to_base64(fig)
    
    return visualizations

def visualize_numeric_by_category(df, target_variable, numeric_cols):
    """
    Visualize numeric features grouped by a categorical target.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        target_variable (str): Name of the target variable
        numeric_cols (list): List of numeric column names
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    visualizations = {}
    
    for col in numeric_cols:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create boxplot
        sns.boxplot(
            x=target_variable, 
            y=col, 
            data=df,
            palette='viridis',
            ax=ax
        )
        
        # Add swarmplot for small datasets
        if len(df) < 1000:
            sns.swarmplot(
                x=target_variable, 
                y=col, 
                data=df,
                color='black',
                alpha=0.5,
                ax=ax
            )
        
        # Format plot
        plt.title(f'Distribution of {col} by {target_variable}', fontsize=14)
        plt.xlabel(target_variable, fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save to dict
        visualizations[col] = fig_to_base64(fig)
    
    return visualizations

def visualize_categorical_associations(df, target_variable, categorical_cols):
    """
    Visualize associations between categorical features and a categorical target.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        target_variable (str): Name of the target variable
        categorical_cols (list): List of categorical column names
    
    Returns:
        dict: Dictionary with visualization base64 images
    """
    visualizations = {}
    
    for col in categorical_cols:
        # Skip if too many categories
        if df[col].nunique() > 10:
            continue
        
        # Create crosstab
        cross_tab = pd.crosstab(
            df[col], 
            df[target_variable],
            normalize='index'
        ) * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(
            cross_tab, 
            annot=True, 
            fmt='.1f',
            cmap='viridis',
            ax=ax
        )
        
        # Format plot
        plt.title(f'Association between {col} and {target_variable} (%)', fontsize=14)
        plt.xlabel(target_variable, fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.tight_layout()
        
        # Save to dict
        visualizations[col] = fig_to_base64(fig)
    
    return visualizations 