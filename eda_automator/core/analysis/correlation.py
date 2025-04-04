"""
Correlation analysis functions for EDA Automator

This module provides functions for analyzing correlations between variables in datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def correlate_features(
    df: pd.DataFrame,
    method: str = 'pearson',
    threshold: float = 0.7,
    include_categorical: bool = True,
    target_variable: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze correlations between variables in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze
    method : str, default 'pearson'
        Correlation method to use ('pearson', 'spearman', or 'kendall')
    threshold : float, default 0.7
        Threshold for significant correlation (absolute value)
    include_categorical : bool, default True
        Whether to include categorical variables (using encoding)
    target_variable : str, optional
        Name of target variable to focus correlation analysis on
    exclude_columns : list of str, optional
        Columns to exclude from analysis
        
    Returns
    -------
    dict
        Dictionary of correlation analysis results
    """
    logger.info(f"Analyzing correlations using method={method}, threshold={threshold}")
    
    # Initialize results
    results = {
        'method': method,
        'threshold': threshold,
        'correlation_matrix': None,
        'strong_correlations': [],
        'target_correlations': [],
        'features_to_remove': [],
        'summary': {},
        'recommendations': []
    }
    
    # Validate method
    valid_methods = ['pearson', 'spearman', 'kendall']
    if method not in valid_methods:
        logger.warning(f"Invalid correlation method: {method}. Falling back to 'pearson'")
        method = 'pearson'
        results['method'] = method
    
    # Create a copy of the dataframe for processing
    processed_df = df.copy()
    
    # Exclude specified columns
    if exclude_columns:
        for col in exclude_columns:
            if col in processed_df.columns:
                processed_df = processed_df.drop(columns=[col])
    
    # Get numeric columns
    numeric_columns = processed_df.select_dtypes(include=['number']).columns.tolist()
    
    # Handle categorical columns if requested
    if include_categorical:
        categorical_columns = processed_df.select_dtypes(include=['category', 'object']).columns.tolist()
        
        # Only encode categorical columns with reasonable cardinality
        for col in categorical_columns:
            n_unique = processed_df[col].nunique()
            if n_unique < 20 and n_unique > 1:  # Skip high cardinality or constant columns
                # One-hot encode categorical variables
                try:
                    # Get dummies and add original column name as prefix
                    dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
                    
                    # Add encoded columns to dataframe
                    processed_df = pd.concat([processed_df, dummies], axis=1)
                except Exception as e:
                    logger.warning(f"Error encoding categorical column {col}: {str(e)}")
    
    # Update numeric columns after potential encoding
    numeric_columns = processed_df.select_dtypes(include=['number']).columns.tolist()
    
    # Handle case with no numeric columns
    if not numeric_columns:
        logger.warning("No numeric columns available for correlation analysis")
        results['recommendations'].append({
            'severity': 'warning',
            'message': 'No numeric columns available for correlation analysis.'
        })
        return results
    
    # Calculate correlation matrix
    try:
        corr_matrix = processed_df[numeric_columns].corr(method=method)
        
        # Replace NaN with 0 for cleaner handling
        corr_matrix = corr_matrix.fillna(0)
        
        # Store as dictionary for JSON serialization
        results['correlation_matrix'] = corr_matrix.to_dict()
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {str(e)}")
        results['recommendations'].append({
            'severity': 'error',
            'message': f'Error in correlation calculation: {str(e)}'
        })
        return results
    
    # Find strong correlations
    strong_correlations = []
    
    # Iterate through the correlation matrix (upper triangle only)
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })
    
    # Sort by absolute correlation value
    strong_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    results['strong_correlations'] = strong_correlations
    
    # Target variable analysis
    if target_variable and target_variable in corr_matrix.columns:
        # Get correlations with target
        target_correlations = []
        
        for col in corr_matrix.columns:
            if col != target_variable:
                corr_value = corr_matrix.loc[target_variable, col]
                
                target_correlations.append({
                    'feature': col,
                    'correlation': corr_value,
                    'abs_correlation': abs(corr_value)
                })
        
        # Sort by absolute correlation value
        target_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        results['target_correlations'] = target_correlations
        
        # Identify important features for target
        important_features = [item['feature'] for item in target_correlations 
                             if item['abs_correlation'] >= threshold]
        
        results['important_features'] = important_features
    
    # Identify features to remove (highly correlated with others)
    features_to_remove = _identify_redundant_features(corr_matrix, threshold=threshold)
    results['features_to_remove'] = features_to_remove
    
    # Summary statistics
    results['summary'] = {
        'num_variables': len(corr_matrix.columns),
        'num_strong_correlations': len(strong_correlations),
        'avg_correlation': np.mean([abs(x) for x in corr_matrix.values.flatten() if x != 1.0]),
        'max_correlation': max([abs(x) for x in corr_matrix.values.flatten() if x != 1.0], default=0),
        'num_redundant_features': len(features_to_remove)
    }
    
    # Generate recommendations
    # 1. For multicollinearity
    if strong_correlations:
        top_corr = strong_correlations[0]
        results['recommendations'].append({
            'severity': 'medium' if abs(top_corr['correlation']) > 0.9 else 'info',
            'message': f"Strong correlation ({top_corr['correlation']:.2f}) detected between {top_corr['feature1']} and {top_corr['feature2']}. Consider removing one to reduce multicollinearity."
        })
    
    # 2. For feature selection
    if features_to_remove:
        features_str = ", ".join(features_to_remove[:5])
        if len(features_to_remove) > 5:
            features_str += f", and {len(features_to_remove) - 5} more"
            
        results['recommendations'].append({
            'severity': 'medium' if len(features_to_remove) > 2 else 'info',
            'message': f"Consider removing redundant features for dimensionality reduction: {features_str}"
        })
    
    # 3. For target variable
    if target_variable and target_variable in corr_matrix.columns:
        if not results.get('important_features'):
            results['recommendations'].append({
                'severity': 'medium',
                'message': f"No features show strong correlation (>= {threshold}) with target {target_variable}. Consider feature engineering or different modeling approaches."
            })
        else:
            top_feature = results['target_correlations'][0]['feature']
            top_corr = results['target_correlations'][0]['correlation']
            
            results['recommendations'].append({
                'severity': 'info',
                'message': f"Feature {top_feature} has strongest correlation ({top_corr:.2f}) with target {target_variable}."
            })
    
    logger.info(f"Correlation analysis completed. Found {len(strong_correlations)} strong correlations.")
    return results

def _identify_redundant_features(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7
) -> List[str]:
    """
    Identify redundant features that can be removed to reduce multicollinearity.
    
    Uses a greedy algorithm to identify features that have high correlation with others.
    
    Parameters
    ----------
    corr_matrix : pandas.DataFrame
        Correlation matrix
    threshold : float, default 0.7
        Threshold for significant correlation
        
    Returns
    -------
    list
        List of feature names that could be removed
    """
    # Get feature names
    columns = corr_matrix.columns.tolist()
    
    # Initialize set of columns to drop
    columns_to_drop = set()
    
    # For each column, check correlation with other columns
    for i in range(len(columns)):
        # Skip if column already marked for removal
        if columns[i] in columns_to_drop:
            continue
        
        # Find correlated features
        for j in range(i + 1, len(columns)):
            # Skip if column already marked for removal
            if columns[j] in columns_to_drop:
                continue
            
            # Check correlation
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Mark one of the columns for removal
                # Prefer to keep the column with higher average correlation with other features
                col1_mean_corr = corr_matrix[columns[i]].abs().mean()
                col2_mean_corr = corr_matrix[columns[j]].abs().mean()
                
                # Drop the column with lower mean correlation
                if col1_mean_corr < col2_mean_corr:
                    columns_to_drop.add(columns[i])
                    break  # Move to next column i
                else:
                    columns_to_drop.add(columns[j])
    
    return list(columns_to_drop)

def compute_cramers_v(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Compute Cramer's V correlation for categorical variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze
    categorical_columns : list of str, optional
        Specific categorical columns to analyze. If None, all object and category columns are used
    threshold : float, default 0.3
        Threshold for significant correlation
        
    Returns
    -------
    dict
        Dictionary of Cramer's V correlation results
    """
    from scipy.stats import chi2_contingency
    
    logger.info("Computing Cramer's V correlations for categorical variables")
    
    # Initialize results
    results = {
        'cramers_v_matrix': {},
        'strong_associations': [],
        'recommendations': []
    }
    
    # Get categorical columns
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        # Validate columns
        invalid_cols = [col for col in categorical_columns if col not in df.columns]
        if invalid_cols:
            raise ValueError(f"Columns not found in DataFrame: {invalid_cols}")
        
        # Filter to categorical only
        categorical_columns = [col for col in categorical_columns 
                              if pd.api.types.is_categorical_dtype(df[col]) or 
                              pd.api.types.is_object_dtype(df[col])]
    
    if not categorical_columns or len(categorical_columns) < 2:
        logger.info("Not enough categorical columns for Cramer's V analysis")
        results['recommendations'].append({
            'severity': 'info',
            'message': 'Not enough categorical columns for association analysis.'
        })
        return results
    
    # Initialize matrix
    cramers_v_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
    
    # Calculate Cramer's V for each pair of categorical variables
    for i, col1 in enumerate(categorical_columns):
        for j, col2 in enumerate(categorical_columns):
            if i <= j:  # Only calculate upper triangle and diagonal
                if i == j:
                    # Diagonal is 1.0
                    cramers_v_matrix.loc[col1, col2] = 1.0
                else:
                    # Calculate Cramer's V
                    try:
                        # Create contingency table
                        contingency = pd.crosstab(df[col1], df[col2])
                        
                        # Calculate Chi-square statistic
                        chi2, p, dof, expected = chi2_contingency(contingency)
                        
                        # Calculate Cramer's V
                        n = contingency.sum().sum()
                        phi2 = chi2 / n
                        r, k = contingency.shape
                        v = np.sqrt(phi2 / min(k-1, r-1))
                        
                        # Store in matrix
                        cramers_v_matrix.loc[col1, col2] = v
                        cramers_v_matrix.loc[col2, col1] = v
                    except Exception as e:
                        logger.warning(f"Error calculating Cramer's V for {col1} and {col2}: {str(e)}")
                        cramers_v_matrix.loc[col1, col2] = 0.0
                        cramers_v_matrix.loc[col2, col1] = 0.0
    
    # Store matrix
    results['cramers_v_matrix'] = cramers_v_matrix.to_dict()
    
    # Find strong associations
    strong_associations = []
    
    for i, col1 in enumerate(categorical_columns):
        for j, col2 in enumerate(categorical_columns):
            if i < j:  # Only upper triangle
                v_value = cramers_v_matrix.loc[col1, col2]
                
                if v_value >= threshold:
                    strong_associations.append({
                        'feature1': col1,
                        'feature2': col2,
                        'cramers_v': v_value
                    })
    
    # Sort by Cramer's V
    strong_associations.sort(key=lambda x: x['cramers_v'], reverse=True)
    
    results['strong_associations'] = strong_associations
    
    # Generate recommendations
    if strong_associations:
        top_assoc = strong_associations[0]
        results['recommendations'].append({
            'severity': 'medium' if top_assoc['cramers_v'] > 0.5 else 'info',
            'message': f"Strong association ({top_assoc['cramers_v']:.2f}) detected between categorical variables {top_assoc['feature1']} and {top_assoc['feature2']}."
        })
    
    logger.info(f"Cramer's V analysis completed. Found {len(strong_associations)} strong associations.")
    return results 