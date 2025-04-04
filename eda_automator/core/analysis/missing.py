"""
Missing values analysis functions for EDA Automator

This module provides functions for analyzing missing values in datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def analyze_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.0,
    include_patterns: bool = True
) -> Dict[str, Any]:
    """
    Analyze missing values in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze
    threshold : float, default 0.0
        Minimum threshold for missing value percentage to include in detailed analysis
    include_patterns : bool, default True
        Whether to analyze missing value patterns
        
    Returns
    -------
    dict
        Dictionary of missing values analysis results
    """
    logger.info("Analyzing missing values")
    
    # Initialize results
    results = {}
    
    # Overall missing values
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    
    results['overall'] = {
        'total_cells': total_cells,
        'total_missing': total_missing,
        'missing_percentage': missing_percentage,
        'complete_rows': (df.notna().all(axis=1)).sum(),
        'complete_rows_percentage': ((df.notna().all(axis=1)).sum() / df.shape[0]) * 100 if df.shape[0] > 0 else 0,
        'complete_columns': (df.notna().all(axis=0)).sum(),
        'complete_columns_percentage': ((df.notna().all(axis=0)).sum() / df.shape[1]) * 100 if df.shape[1] > 0 else 0
    }
    
    # Missing values by column
    missing_by_column = df.isna().sum().sort_values(ascending=False).to_dict()
    missing_percentage_by_column = (df.isna().mean() * 100).sort_values(ascending=False).to_dict()
    
    # Only include columns above threshold for detailed analysis
    columns_above_threshold = [col for col, pct in missing_percentage_by_column.items() 
                              if pct >= threshold]
    
    results['by_column'] = {
        'counts': missing_by_column,
        'percentages': missing_percentage_by_column,
        'columns_above_threshold': columns_above_threshold
    }
    
    # Missing values by row
    missing_by_row_count = df.isna().sum(axis=1)
    missing_by_row_percentage = (df.isna().sum(axis=1) / df.shape[1]) * 100
    
    # Calculate row statistics
    results['by_row'] = {
        'summary': {
            'min': missing_by_row_count.min(),
            'max': missing_by_row_count.max(),
            'mean': missing_by_row_count.mean(),
            'median': missing_by_row_count.median(),
            'min_percentage': missing_by_row_percentage.min(),
            'max_percentage': missing_by_row_percentage.max(),
            'mean_percentage': missing_by_row_percentage.mean(),
            'median_percentage': missing_by_row_percentage.median()
        },
        'distribution': {
            str(count): int((missing_by_row_count == count).sum())
            for count in sorted(missing_by_row_count.unique())
        },
        'high_missing_rows': int((missing_by_row_percentage >= 50).sum()),
        'high_missing_percentage': ((missing_by_row_percentage >= 50).sum() / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    }
    
    # Missing patterns analysis
    if include_patterns and len(columns_above_threshold) > 0:
        # Limit to a reasonable number of patterns to analyze
        pattern_columns = columns_above_threshold
        if len(pattern_columns) > 10:
            pattern_columns = pattern_columns[:10]
            logger.info(f"Limiting pattern analysis to top 10 columns with highest missing percentage")
        
        try:
            # Create missing pattern matrix
            pattern_df = df[pattern_columns].isna()
            patterns = pattern_df.value_counts().reset_index()
            
            # Limit to top 20 patterns
            if len(patterns) > 20:
                patterns = patterns.head(20)
            
            # Convert patterns to readable format
            pattern_list = []
            for _, row in patterns.iterrows():
                pattern_dict = {}
                for i, col in enumerate(pattern_columns):
                    pattern_dict[col] = bool(row[col])
                
                pattern_dict['count'] = int(row.iloc[-1])
                pattern_dict['percentage'] = (row.iloc[-1] / df.shape[0]) * 100
                pattern_list.append(pattern_dict)
            
            results['patterns'] = pattern_list
        except Exception as e:
            logger.warning(f"Error in pattern analysis: {str(e)}")
            results['patterns'] = []
    else:
        results['patterns'] = []
    
    # Correlations between missing values
    if include_patterns and len(columns_above_threshold) >= 2:
        try:
            # Create binary missing matrix (1 if missing, 0 if not)
            missing_df = df[columns_above_threshold].isna().astype(int)
            
            # Calculate correlations between missing indicators
            missing_corr = missing_df.corr()
            
            # Get top correlated pairs (excluding self-correlations)
            corr_pairs = []
            for i, col1 in enumerate(missing_corr.columns):
                for j, col2 in enumerate(missing_corr.columns):
                    if i < j:  # Only include each pair once
                        correlation = missing_corr.iloc[i, j]
                        if abs(correlation) > 0.5:  # Only include strong correlations
                            corr_pairs.append({
                                'column1': col1,
                                'column2': col2,
                                'correlation': correlation,
                                'abs_correlation': abs(correlation)
                            })
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            # Limit to top 10 pairs
            results['missing_correlations'] = corr_pairs[:10]
        except Exception as e:
            logger.warning(f"Error in missing correlation analysis: {str(e)}")
            results['missing_correlations'] = []
    else:
        results['missing_correlations'] = []
    
    # Recommendations
    recommendations = []
    
    # Overall recommendations
    if missing_percentage > 50:
        recommendations.append({
            'severity': 'high',
            'message': 'Dataset has very high missing data (>50%). Consider acquiring more complete data or reducing the scope of analysis.'
        })
    elif missing_percentage > 20:
        recommendations.append({
            'severity': 'medium',
            'message': 'Dataset has substantial missing data (>20%). Consider imputation strategies or removing columns with excessive missing values.'
        })
    
    # Column-specific recommendations
    high_missing_cols = [col for col, pct in missing_percentage_by_column.items() if pct >= 75]
    medium_missing_cols = [col for col, pct in missing_percentage_by_column.items() if 30 <= pct < 75]
    
    if high_missing_cols:
        recommendations.append({
            'severity': 'high',
            'message': f'Consider dropping columns with very high missing values (>75%): {", ".join(high_missing_cols[:5])}{"..." if len(high_missing_cols) > 5 else ""}'
        })
    
    if medium_missing_cols:
        recommendations.append({
            'severity': 'medium',
            'message': f'Consider imputing missing values for columns with moderate missingness (30-75%): {", ".join(medium_missing_cols[:5])}{"..." if len(medium_missing_cols) > 5 else ""}'
        })
    
    # Row-specific recommendations
    if results['by_row']['high_missing_percentage'] > 30:
        recommendations.append({
            'severity': 'medium',
            'message': f'Consider filtering out rows with high missing values (>50% missing), which comprise {results["by_row"]["high_missing_percentage"]:.1f}% of the dataset.'
        })
    
    # Pattern-based recommendations
    if results['missing_correlations'] and results['missing_correlations'][0]['abs_correlation'] > 0.9:
        col1 = results['missing_correlations'][0]['column1']
        col2 = results['missing_correlations'][0]['column2']
        recommendations.append({
            'severity': 'info',
            'message': f'Missing values in {col1} and {col2} are highly correlated (r={results["missing_correlations"][0]["correlation"]:.2f}). They might be missing for the same reason.'
        })
    
    results['recommendations'] = recommendations
    
    logger.info("Missing values analysis completed")
    return results

def suggest_imputation_methods(
    df: pd.DataFrame,
    column: str
) -> Dict[str, Any]:
    """
    Suggest appropriate imputation methods for a column based on its characteristics.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column
    column : str
        Name of the column to analyze
        
    Returns
    -------
    dict
        Dictionary of suggested imputation methods and their rationale
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if df[column].isna().sum() == 0:
        return {'message': 'No missing values in this column'}
    
    logger.info(f"Suggesting imputation methods for column: {column}")
    
    # Initialize results
    results = {
        'column': column,
        'missing_count': df[column].isna().sum(),
        'missing_percentage': (df[column].isna().sum() / len(df)) * 100,
        'suggestions': []
    }
    
    # Get column characteristics
    is_numeric = pd.api.types.is_numeric_dtype(df[column])
    is_categorical = pd.api.types.is_categorical_dtype(df[column])
    is_object = pd.api.types.is_object_dtype(df[column])
    is_datetime = pd.api.types.is_datetime64_dtype(df[column])
    is_boolean = pd.api.types.is_bool_dtype(df[column])
    
    # Analyze distribution for categorical/object columns
    if is_categorical or is_object:
        value_counts = df[column].value_counts(normalize=True)
        most_frequent = value_counts.index[0] if not value_counts.empty else None
        most_frequent_pct = value_counts.iloc[0] * 100 if not value_counts.empty else 0
        is_imbalanced = most_frequent_pct > 80  # One value dominates
        
        if is_imbalanced:
            results['suggestions'].append({
                'method': 'constant',
                'value': str(most_frequent),
                'rationale': f"Replace with most frequent value ('{most_frequent}'), which appears in {most_frequent_pct:.1f}% of non-missing values."
            })
        else:
            results['suggestions'].append({
                'method': 'mode',
                'rationale': 'Replace with the most frequent value (statistical mode)'
            })
            
        results['suggestions'].append({
            'method': 'new_category',
            'rationale': 'Create a new category "Unknown" or "Missing" to explicitly represent missing values'
        })
    
    # Analyze distribution for numeric columns
    elif is_numeric:
        # Check for skewness
        skewness = df[column].skew()
        is_skewed = abs(skewness) > 1
        
        # Check for outliers
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        has_outliers = ((df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))).any()
        
        # Suggest methods
        results['suggestions'].append({
            'method': 'mean',
            'rationale': 'Replace with mean value. Good for normally distributed data without outliers.'
        })
        
        results['suggestions'].append({
            'method': 'median',
            'rationale': 'Replace with median value. Good for skewed data or data with outliers.'
        })
        
        if is_skewed or has_outliers:
            results['suggestions'].append({
                'method': 'median',
                'priority': 'high',
                'rationale': f"Median is preferred over mean due to {'skewness' if is_skewed else ''}{' and ' if is_skewed and has_outliers else ''}{'outliers' if has_outliers else ''}"
            })
        else:
            results['suggestions'].append({
                'method': 'mean',
                'priority': 'high',
                'rationale': 'Mean is appropriate as the data is relatively symmetric without significant outliers'
            })
            
        # Advanced methods
        results['suggestions'].append({
            'method': 'knn',
            'rationale': 'K-Nearest Neighbors imputation uses similar records to predict missing values'
        })
        
        results['suggestions'].append({
            'method': 'regression',
            'rationale': 'Use regression models to predict missing values based on other columns'
        })
    
    # Analyze datetime columns
    elif is_datetime:
        results['suggestions'].append({
            'method': 'forward_fill',
            'rationale': 'Propagate last valid observation forward (good for time series data)'
        })
        
        results['suggestions'].append({
            'method': 'backward_fill',
            'rationale': 'Use next valid observation to fill gap (alternative for time series data)'
        })
        
        results['suggestions'].append({
            'method': 'median_date',
            'rationale': 'Replace with median date'
        })
    
    # Boolean columns
    elif is_boolean:
        true_count = df[column].sum()
        false_count = len(df[column].dropna()) - true_count
        majority_value = True if true_count > false_count else False
        majority_pct = max(true_count, false_count) / len(df[column].dropna()) * 100
        
        results['suggestions'].append({
            'method': 'constant',
            'value': str(majority_value),
            'rationale': f"Replace with majority value ({majority_value}), which appears in {majority_pct:.1f}% of non-missing values."
        })
        
        results['suggestions'].append({
            'method': 'random',
            'rationale': 'Replace with random values based on the observed distribution of True/False'
        })
    
    # General suggestions for all types
    if results['missing_percentage'] > 80:
        results['recommendations'] = [{
            'severity': 'high',
            'message': f"Column has {results['missing_percentage']:.1f}% missing values. Consider dropping this column."
        }]
    elif results['missing_percentage'] > 50:
        results['recommendations'] = [{
            'severity': 'medium',
            'message': f"Column has {results['missing_percentage']:.1f}% missing values. Imputation might introduce bias."
        }]
    else:
        results['recommendations'] = [{
            'severity': 'info',
            'message': f"Column has {results['missing_percentage']:.1f}% missing values. Standard imputation methods should work well."
        }]
    
    logger.info(f"Imputation suggestions completed for: {column}")
    return results 