"""
Basic data analysis functions for EDA Automator

This module provides functions for basic dataset analysis, including
data profiling, statistics, and data type analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def analyze_data(
    df: pd.DataFrame,
    include_preview: bool = True,
    include_stats: bool = True,
    include_dtypes: bool = True
) -> Dict[str, Any]:
    """
    Perform basic analysis on a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze
    include_preview : bool, default True
        Whether to include a preview of the data
    include_stats : bool, default True
        Whether to include basic statistics
    include_dtypes : bool, default True
        Whether to include data type information
        
    Returns
    -------
    dict
        Dictionary of analysis results
    """
    logger.info("Running basic data analysis")
    
    # Initialize results dictionary
    results = {}
    
    # Get basic information
    results['shape'] = {'rows': df.shape[0], 'columns': df.shape[1]}
    results['memory_usage'] = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    # Data types analysis
    if include_dtypes:
        results['dtypes'] = df.dtypes.astype(str).to_dict()
        
        # Identify column types
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        boolean_columns = df.select_dtypes(include=['bool']).columns.tolist()
        
        results['column_types'] = {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'datetime': datetime_columns,
            'boolean': boolean_columns
        }
    
    # Basic statistics
    if include_stats:
        # Numeric statistics
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            results['numeric_stats'] = df[numeric_columns].describe().to_dict()
            
            # Additional statistics not included in describe()
            skew = df[numeric_columns].skew().to_dict()
            kurtosis = df[numeric_columns].kurtosis().to_dict()
            
            results['numeric_stats_extra'] = {
                'skew': skew,
                'kurtosis': kurtosis
            }
        
        # Categorical statistics
        categorical_columns = df.select_dtypes(include=['category', 'object']).columns
        if len(categorical_columns) > 0:
            category_counts = {}
            for col in categorical_columns:
                value_counts = df[col].value_counts().head(10).to_dict()  # Top 10 categories
                unique_count = df[col].nunique()
                category_counts[col] = {
                    'value_counts': value_counts,
                    'unique_count': unique_count
                }
            
            results['category_counts'] = category_counts
        
        # Missing values summary
        missing_counts = df.isna().sum().to_dict()
        missing_percentages = (df.isna().mean() * 100).to_dict()
        
        results['missing_summary'] = {
            'counts': missing_counts,
            'percentages': missing_percentages,
            'total_missing': df.isna().sum().sum(),
            'total_missing_percentage': (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        }
    
    # Data preview
    if include_preview:
        results['preview'] = {
            'head': df.head(5).to_dict('records'),
            'tail': df.tail(5).to_dict('records')
        }
    
    logger.info("Basic data analysis completed")
    return results

def analyze_column(
    df: pd.DataFrame,
    column: str
) -> Dict[str, Any]:
    """
    Analyze a single column in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column
    column : str
        Name of the column to analyze
        
    Returns
    -------
    dict
        Dictionary of column analysis results
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    logger.info(f"Analyzing column: {column}")
    
    # Initialize results
    results = {
        'name': column,
        'dtype': str(df[column].dtype)
    }
    
    # Basic information
    results['count'] = len(df[column])
    results['unique_count'] = df[column].nunique()
    results['missing_count'] = df[column].isna().sum()
    results['missing_percentage'] = (df[column].isna().sum() / len(df[column])) * 100
    
    # Type-specific analysis
    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric column
        results['type'] = 'numeric'
        results['stats'] = {
            'min': df[column].min(),
            'max': df[column].max(),
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'skew': df[column].skew(),
            'kurtosis': df[column].kurtosis(),
            'q1': df[column].quantile(0.25),
            'q3': df[column].quantile(0.75),
            'iqr': df[column].quantile(0.75) - df[column].quantile(0.25)
        }
        
        # Check if likely integer
        if df[column].dropna().apply(lambda x: x.is_integer()).all():
            results['likely_integer'] = True
        else:
            results['likely_integer'] = False
            
        # Check if likely binary (0-1)
        unique_values = df[column].dropna().unique()
        if len(unique_values) == 2 and set(unique_values) == {0, 1}:
            results['likely_binary'] = True
        else:
            results['likely_binary'] = False
    
    elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
        # Categorical or string column
        results['type'] = 'categorical'
        
        # Get value counts
        value_counts = df[column].value_counts().head(10).to_dict()
        results['value_counts'] = value_counts
        
        # Calculate frequency percentages
        total_non_null = len(df[column].dropna())
        frequency_percentages = {k: (v / total_non_null) * 100 for k, v in value_counts.items()}
        results['frequency_percentages'] = frequency_percentages
        
        # Check if likely categorical (few unique values)
        unique_ratio = df[column].nunique() / len(df[column].dropna())
        results['unique_ratio'] = unique_ratio
        results['likely_categorical'] = unique_ratio < 0.05  # Less than 5% unique values
        
        # Check if likely boolean/binary
        unique_values = set(df[column].dropna().unique())
        if len(unique_values) == 2:
            if unique_values == {'True', 'False'} or unique_values == {True, False} or \
               unique_values == {'Yes', 'No'} or unique_values == {'Y', 'N'} or \
               unique_values == {0, 1} or unique_values == {'0', '1'}:
                results['likely_boolean'] = True
            else:
                results['likely_boolean'] = False
        else:
            results['likely_boolean'] = False
    
    elif pd.api.types.is_datetime64_dtype(df[column]):
        # Datetime column
        results['type'] = 'datetime'
        
        non_null_dates = df[column].dropna()
        if len(non_null_dates) > 0:
            results['stats'] = {
                'min': non_null_dates.min(),
                'max': non_null_dates.max(),
                'range_days': (non_null_dates.max() - non_null_dates.min()).days
            }
            
            # Extract distribution of dates
            if len(non_null_dates) >= 20:  # Only if enough data
                try:
                    date_counts = df[column].dt.date.value_counts().sort_index()
                    year_counts = df[column].dt.year.value_counts().sort_index()
                    month_counts = df[column].dt.month.value_counts().sort_index()
                    weekday_counts = df[column].dt.weekday.value_counts().sort_index()
                    
                    results['date_distribution'] = {
                        'years': {str(k): int(v) for k, v in year_counts.items()},
                        'months': {str(k): int(v) for k, v in month_counts.items()},
                        'weekdays': {str(k): int(v) for k, v in weekday_counts.items()}
                    }
                except:
                    pass  # Skip date distribution if it fails
    
    elif pd.api.types.is_bool_dtype(df[column]):
        # Boolean column
        results['type'] = 'boolean'
        
        # Get counts
        value_counts = df[column].value_counts().to_dict()
        results['value_counts'] = value_counts
        
        # Calculate percentages
        total_non_null = len(df[column].dropna())
        if total_non_null > 0:
            true_percentage = (value_counts.get(True, 0) / total_non_null) * 100
            false_percentage = (value_counts.get(False, 0) / total_non_null) * 100
            
            results['true_percentage'] = true_percentage
            results['false_percentage'] = false_percentage
    
    else:
        # Other types
        results['type'] = 'other'
    
    logger.info(f"Column analysis completed for: {column}")
    return results 