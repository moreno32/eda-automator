"""
Outlier detection functions for EDA Automator

This module provides functions for detecting and analyzing outliers in datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def detect_outliers(
    df: pd.DataFrame,
    method: str = 'z-score',
    threshold: float = 3.0,
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Detect outliers in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze
    method : str, default 'z-score'
        Method to use for outlier detection. Options:
        - 'z-score': Use z-score (standard deviations from mean)
        - 'iqr': Use interquartile range method
        - 'isolation-forest': Use isolation forest algorithm (requires sklearn)
    threshold : float, default 3.0
        Threshold for outlier detection:
        - For z-score: number of standard deviations
        - For IQR: multiplier for IQR (typical values: 1.5 or 3.0)
        - For isolation-forest: contamination parameter
    columns : list of str, optional
        Specific columns to analyze. If None, all numeric columns are analyzed
        
    Returns
    -------
    dict
        Dictionary of outlier analysis results
    """
    logger.info(f"Detecting outliers using method={method}, threshold={threshold}")
    
    # Initialize results
    results = {
        'method': method,
        'threshold': threshold,
        'outliers_by_column': {},
        'summary': {},
        'recommendations': []
    }
    
    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter columns if specified
    if columns is not None:
        # Validate specified columns
        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            raise ValueError(f"Columns not found in DataFrame: {invalid_columns}")
            
        # Filter to specified numeric columns
        numeric_columns = [col for col in columns if col in numeric_columns]
        
        if not numeric_columns:
            logger.warning("No numeric columns found for outlier detection")
            results['recommendations'].append({
                'severity': 'warning',
                'message': 'No numeric columns available for outlier detection.'
            })
            return results
    
    # Detect outliers for each column
    for column in numeric_columns:
        # Skip columns with too few values
        if df[column].count() < 5:
            logger.info(f"Skipping column {column} - too few values for outlier detection")
            continue
        
        # Skip boolean-like columns
        unique_values = df[column].dropna().unique()
        if len(unique_values) <= 2:
            logger.info(f"Skipping column {column} - binary/boolean column")
            continue
        
        # Get column data without NaN values
        col_data = df[column].dropna()
        
        # Detect outliers based on method
        outlier_indices = []
        
        if method == 'z-score':
            # Z-score method
            mean = col_data.mean()
            std = col_data.std()
            z_scores = np.abs((col_data - mean) / std)
            outlier_indices = np.where(z_scores > threshold)[0]
            
            # Get the indices in the original dataframe
            outlier_indices = col_data.iloc[outlier_indices].index.tolist()
            
        elif method == 'iqr':
            # IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_indices = outliers.index.tolist()
            
        elif method == 'isolation-forest':
            # Isolation Forest method (requires sklearn)
            try:
                from sklearn.ensemble import IsolationForest
            except ImportError:
                logger.error("scikit-learn is required for isolation-forest method")
                results['recommendations'].append({
                    'severity': 'error',
                    'message': 'scikit-learn is required for isolation-forest method. Install it with "pip install scikit-learn".'
                })
                method = 'iqr'  # Fall back to IQR method
                logger.info("Falling back to IQR method")
                
                # Recalculate using IQR method
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_indices = outliers.index.tolist()
            else:
                # Use Isolation Forest if sklearn is available
                contamination = min(0.1, max(0.01, threshold / 10))  # Convert threshold to contamination parameter
                
                # Reshape data for sklearn
                X = col_data.values.reshape(-1, 1)
                
                # Train model
                model = IsolationForest(contamination=contamination, random_state=42)
                yhat = model.fit_predict(X)
                
                # Get outlier mask (-1 for outliers, 1 for inliers)
                mask = yhat == -1
                
                # Get the indices in the original dataframe
                outlier_indices = col_data.iloc[np.where(mask)[0]].index.tolist()
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Get outlier values
        outlier_values = df.loc[outlier_indices, column].tolist() if outlier_indices else []
        
        # Store results for this column
        if outlier_indices:
            # Calculate outlier statistics
            non_outlier_values = df.loc[~df.index.isin(outlier_indices), column].dropna()
            outlier_values_series = df.loc[outlier_indices, column]
            
            statistics = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(col_data)) * 100,
                'min': float(outlier_values_series.min()),
                'max': float(outlier_values_series.max()),
                'mean': float(outlier_values_series.mean()),
                'non_outlier_min': float(non_outlier_values.min()),
                'non_outlier_max': float(non_outlier_values.max()),
                'non_outlier_mean': float(non_outlier_values.mean())
            }
            
            # Store detailed information about the first few outliers
            max_examples = 10
            examples = []
            for idx in outlier_indices[:max_examples]:
                examples.append({
                    'index': int(idx),
                    'value': float(df.loc[idx, column])
                })
            
            results['outliers_by_column'][column] = {
                'statistics': statistics,
                'examples': examples,
                'all_indices': [int(idx) for idx in outlier_indices]
            }
        else:
            # No outliers found
            results['outliers_by_column'][column] = {
                'statistics': {
                    'count': 0,
                    'percentage': 0.0
                },
                'examples': [],
                'all_indices': []
            }
    
    # Compute summary statistics
    total_outliers = sum(results['outliers_by_column'][col]['statistics']['count'] 
                         for col in results['outliers_by_column'])
    
    columns_with_outliers = [col for col in results['outliers_by_column'] 
                            if results['outliers_by_column'][col]['statistics']['count'] > 0]
    
    results['summary'] = {
        'total_outliers': total_outliers,
        'columns_analyzed': len(results['outliers_by_column']),
        'columns_with_outliers': len(columns_with_outliers),
        'columns_with_outliers_list': columns_with_outliers
    }
    
    # Generate recommendations
    high_outlier_columns = []
    moderate_outlier_columns = []
    
    for col in results['outliers_by_column']:
        outlier_pct = results['outliers_by_column'][col]['statistics'].get('percentage', 0)
        
        if outlier_pct >= 10:
            high_outlier_columns.append((col, outlier_pct))
        elif outlier_pct >= 5:
            moderate_outlier_columns.append((col, outlier_pct))
    
    # Sort columns by outlier percentage
    high_outlier_columns.sort(key=lambda x: x[1], reverse=True)
    moderate_outlier_columns.sort(key=lambda x: x[1], reverse=True)
    
    # Add recommendations
    if high_outlier_columns:
        cols_str = ", ".join([f"{col} ({pct:.1f}%)" for col, pct in high_outlier_columns[:3]])
        if len(high_outlier_columns) > 3:
            cols_str += f", and {len(high_outlier_columns) - 3} more"
            
        results['recommendations'].append({
            'severity': 'high',
            'message': f"High outlier percentage in columns: {cols_str}. Consider reviewing data collection or using robust methods."
        })
    
    if moderate_outlier_columns:
        cols_str = ", ".join([f"{col} ({pct:.1f}%)" for col, pct in moderate_outlier_columns[:3]])
        if len(moderate_outlier_columns) > 3:
            cols_str += f", and {len(moderate_outlier_columns) - 3} more"
            
        results['recommendations'].append({
            'severity': 'medium',
            'message': f"Moderate outlier percentage in columns: {cols_str}. Consider transforming data or using robust methods."
        })
    
    if not high_outlier_columns and not moderate_outlier_columns and results['summary']['total_outliers'] > 0:
        results['recommendations'].append({
            'severity': 'info',
            'message': f"Low percentage of outliers detected across {len(columns_with_outliers)} columns."
        })
    
    if results['summary']['total_outliers'] == 0:
        results['recommendations'].append({
            'severity': 'info',
            'message': f"No outliers detected using {method} method with threshold {threshold}."
        })
    
    logger.info(f"Outlier detection completed. Found {results['summary']['total_outliers']} outliers in {results['summary']['columns_with_outliers']} columns.")
    return results

def handle_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = 'winsorize',
    threshold: float = 3.0,
    detection_method: str = 'z-score'
) -> pd.DataFrame:
    """
    Handle outliers in a single column of a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column
    column : str
        Name of the column to process
    method : str, default 'winsorize'
        Method to handle outliers:
        - 'winsorize': Cap values at the threshold
        - 'trim': Remove outlier rows
        - 'mean': Replace with mean
        - 'median': Replace with median
        - 'missing': Replace with NaN
    threshold : float, default 3.0
        Threshold for outlier detection
    detection_method : str, default 'z-score'
        Method to use for outlier detection ('z-score' or 'iqr')
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with handled outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Check if the column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' is not numeric")
    
    logger.info(f"Handling outliers in column '{column}' using method '{method}'")
    
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Get column data without NaN values
    col_data = df_copy[column].dropna()
    
    # Detect outliers
    outlier_mask = pd.Series(False, index=col_data.index)
    
    if detection_method == 'z-score':
        # Z-score method
        mean = col_data.mean()
        std = col_data.std()
        z_scores = np.abs((col_data - mean) / std)
        outlier_mask = z_scores > threshold
    elif detection_method == 'iqr':
        # IQR method
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
    else:
        raise ValueError(f"Unknown outlier detection method: {detection_method}")
    
    # Get indices of outliers
    outlier_indices = col_data.index[outlier_mask]
    
    # Handle outliers based on method
    if method == 'winsorize':
        if detection_method == 'z-score':
            # Cap values at mean +/- threshold * std
            upper_cap = mean + threshold * std
            lower_cap = mean - threshold * std
            
            # Cap values
            df_copy.loc[df_copy[column] > upper_cap, column] = upper_cap
            df_copy.loc[df_copy[column] < lower_cap, column] = lower_cap
            
        elif detection_method == 'iqr':
            # Cap values at bounds
            df_copy.loc[df_copy[column] > upper_bound, column] = upper_bound
            df_copy.loc[df_copy[column] < lower_bound, column] = lower_bound
    
    elif method == 'trim':
        # Remove rows with outliers
        df_copy = df_copy.drop(outlier_indices)
    
    elif method == 'mean':
        # Replace with mean
        df_copy.loc[outlier_indices, column] = col_data[~outlier_mask].mean()
    
    elif method == 'median':
        # Replace with median
        df_copy.loc[outlier_indices, column] = col_data[~outlier_mask].median()
    
    elif method == 'missing':
        # Replace with NaN
        df_copy.loc[outlier_indices, column] = np.nan
    
    else:
        raise ValueError(f"Unknown outlier handling method: {method}")
    
    logger.info(f"Handled {len(outlier_indices)} outliers in column '{column}'")
    return df_copy 