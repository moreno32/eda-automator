"""
Target variable analysis functions for EDA Automator

This module provides functions for analyzing relationships between features and a target variable.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
from scipy import stats

from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def analyze_target_relationship(
    df: pd.DataFrame,
    target_variable: str,
    features: Optional[List[str]] = None,
    max_categories: int = 20,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Analyze relationships between features and a target variable.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze
    target_variable : str
        Name of the target variable
    features : list of str, optional
        List of features to analyze. If None, all suitable columns except
        the target are analyzed
    max_categories : int, default 20
        Maximum number of categories to consider for categorical variables
    significance_level : float, default 0.05
        Significance level for statistical tests
        
    Returns
    -------
    dict
        Dictionary of target analysis results
    """
    logger.info(f"Analyzing relationships with target variable: {target_variable}")
    
    # Initialize results
    results = {
        'target_variable': target_variable,
        'target_type': None,
        'feature_relationships': {},
        'top_features': [],
        'summary': {},
        'recommendations': []
    }
    
    # Check if target variable exists
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in DataFrame")
    
    # Determine target type
    target_is_numeric = pd.api.types.is_numeric_dtype(df[target_variable])
    target_is_categorical = pd.api.types.is_categorical_dtype(df[target_variable]) or pd.api.types.is_object_dtype(df[target_variable])
    target_is_boolean = pd.api.types.is_bool_dtype(df[target_variable])
    
    # For numeric targets with few unique values, treat as categorical
    if target_is_numeric and df[target_variable].nunique() <= 10:
        logger.info(f"Target variable '{target_variable}' is numeric but has few unique values. Treating as categorical.")
        target_is_categorical = True
    
    # Set target type
    if target_is_boolean or (target_is_numeric and df[target_variable].nunique() <= 2):
        results['target_type'] = 'binary'
    elif target_is_categorical:
        results['target_type'] = 'categorical'
    elif target_is_numeric:
        results['target_type'] = 'numeric'
    else:
        results['target_type'] = 'other'
    
    # Process target and features
    df_copy = df.copy()
    
    # Convert target to appropriate type
    if results['target_type'] == 'binary':
        if not pd.api.types.is_bool_dtype(df_copy[target_variable]):
            # Try to convert to binary
            unique_values = df_copy[target_variable].dropna().unique()
            if len(unique_values) <= 2:
                # Use the first unique value as 0 and others as 1
                true_val = unique_values[0]
                df_copy[target_variable] = df_copy[target_variable].apply(lambda x: 0 if x == true_val else 1)
    
    # Get features to analyze
    if features is not None:
        # Validate specified features
        invalid_features = [col for col in features if col not in df.columns]
        if invalid_features:
            raise ValueError(f"Features not found in DataFrame: {invalid_features}")
        analyze_features = [f for f in features if f != target_variable]
    else:
        analyze_features = [col for col in df.columns if col != target_variable]
    
    # Identify feature types
    numeric_features = [col for col in analyze_features 
                       if pd.api.types.is_numeric_dtype(df[col]) and 
                       col in df.columns]
    
    categorical_features = [col for col in analyze_features 
                          if (pd.api.types.is_categorical_dtype(df[col]) or 
                             pd.api.types.is_object_dtype(df[col]) or
                             pd.api.types.is_bool_dtype(df[col])) and 
                          col in df.columns]
    
    # For each feature, analyze relationship with target
    feature_importance = []
    
    # Case 1: Numeric target
    if results['target_type'] == 'numeric':
        # Analyze numeric features
        for col in numeric_features:
            try:
                # Skip if too many missing values
                if df[col].isna().mean() > 0.5 or df[target_variable].isna().mean() > 0.5:
                    continue
                
                # Get non-missing pairs
                valid_data = df[[col, target_variable]].dropna()
                
                # Skip if too few values
                if len(valid_data) < 10:
                    continue
                
                # Calculate correlation
                pearson_corr, pearson_p = stats.pearsonr(valid_data[col], valid_data[target_variable])
                spearman_corr, spearman_p = stats.spearmanr(valid_data[col], valid_data[target_variable])
                
                # Store results
                feature_result = {
                    'feature_type': 'numeric',
                    'correlation': {
                        'pearson': {
                            'coefficient': float(pearson_corr),
                            'p_value': float(pearson_p),
                            'significant': pearson_p < significance_level
                        },
                        'spearman': {
                            'coefficient': float(spearman_corr),
                            'p_value': float(spearman_p),
                            'significant': spearman_p < significance_level
                        }
                    }
                }
                
                # Determine linearity and strength
                abs_diff = abs(abs(pearson_corr) - abs(spearman_corr))
                
                if abs_diff > 0.1:
                    feature_result['relationship_type'] = 'non_linear'
                else:
                    feature_result['relationship_type'] = 'linear'
                
                if abs(pearson_corr) > 0.7 or abs(spearman_corr) > 0.7:
                    feature_result['relationship_strength'] = 'strong'
                elif abs(pearson_corr) > 0.3 or abs(spearman_corr) > 0.3:
                    feature_result['relationship_strength'] = 'moderate'
                else:
                    feature_result['relationship_strength'] = 'weak'
                
                # Store feature result
                results['feature_relationships'][col] = feature_result
                
                # Add to feature importance list
                importance = max(abs(pearson_corr), abs(spearman_corr))
                feature_importance.append((col, importance, 'correlation'))
            
            except Exception as e:
                logger.warning(f"Error analyzing relationship between '{col}' and '{target_variable}': {str(e)}")
        
        # Analyze categorical features
        for col in categorical_features:
            try:
                # Skip if too many categories
                if df[col].nunique() > max_categories:
                    continue
                
                # Skip if too many missing values
                if df[col].isna().mean() > 0.5 or df[target_variable].isna().mean() > 0.5:
                    continue
                
                # Get non-missing pairs
                valid_data = df[[col, target_variable]].dropna()
                
                # Skip if too few values
                if len(valid_data) < 10:
                    continue
                
                # Calculate target stats by category
                category_stats = {}
                category_groups = []
                
                for category in valid_data[col].unique():
                    subset = valid_data[valid_data[col] == category][target_variable]
                    
                    if len(subset) > 0:
                        category_stats[str(category)] = {
                            'count': int(len(subset)),
                            'mean': float(subset.mean()),
                            'median': float(subset.median()),
                            'std': float(subset.std()) if len(subset) > 1 else 0.0,
                            'min': float(subset.min()),
                            'max': float(subset.max())
                        }
                        
                        category_groups.append(subset.values)
                
                # Perform ANOVA to test for significant differences
                if len(category_groups) >= 2 and all(len(g) > 0 for g in category_groups):
                    f_statistic, p_value = stats.f_oneway(*category_groups)
                    
                    anova_result = {
                        'f_statistic': float(f_statistic),
                        'p_value': float(p_value),
                        'significant': p_value < significance_level
                    }
                    
                    # Store results
                    feature_result = {
                        'feature_type': 'categorical',
                        'category_stats': category_stats,
                        'anova': anova_result
                    }
                    
                    # Determine strength of relationship
                    # Calculate effect size (Eta squared)
                    num_categories = len(category_groups)
                    total_samples = sum(len(g) for g in category_groups)
                    
                    eta_squared = (f_statistic * (num_categories - 1)) / (f_statistic * (num_categories - 1) + (total_samples - num_categories))
                    feature_result['eta_squared'] = float(eta_squared)
                    
                    if eta_squared > 0.14:
                        feature_result['relationship_strength'] = 'strong'
                    elif eta_squared > 0.06:
                        feature_result['relationship_strength'] = 'moderate'
                    else:
                        feature_result['relationship_strength'] = 'weak'
                    
                    # Store feature result
                    results['feature_relationships'][col] = feature_result
                    
                    # Add to feature importance list (only if significant)
                    if p_value < significance_level:
                        feature_importance.append((col, eta_squared, 'anova'))
            
            except Exception as e:
                logger.warning(f"Error analyzing relationship between '{col}' and '{target_variable}': {str(e)}")
    
    # Case 2: Binary/Categorical target
    elif results['target_type'] in ['binary', 'categorical']:
        # Analyze numeric features
        for col in numeric_features:
            try:
                # Skip if too many missing values
                if df[col].isna().mean() > 0.5 or df[target_variable].isna().mean() > 0.5:
                    continue
                
                # Get non-missing pairs
                valid_data = df[[col, target_variable]].dropna()
                
                # Skip if too few values
                if len(valid_data) < 10:
                    continue
                
                # Different tests based on target type
                if results['target_type'] == 'binary':
                    # Two-sample t-test for binary targets
                    group0 = valid_data[valid_data[target_variable] == 0][col].values
                    group1 = valid_data[valid_data[target_variable] == 1][col].values
                    
                    if len(group0) > 0 and len(group1) > 0:
                        t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)
                        
                        # Store results
                        feature_result = {
                            'feature_type': 'numeric',
                            't_test': {
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': p_value < significance_level
                            },
                            'group_stats': {
                                '0': {
                                    'count': int(len(group0)),
                                    'mean': float(np.mean(group0)),
                                    'std': float(np.std(group0)),
                                    'min': float(np.min(group0)),
                                    'max': float(np.max(group0))
                                },
                                '1': {
                                    'count': int(len(group1)),
                                    'mean': float(np.mean(group1)),
                                    'std': float(np.std(group1)),
                                    'min': float(np.min(group1)),
                                    'max': float(np.max(group1))
                                }
                            }
                        }
                        
                        # Calculate effect size (Cohen's d)
                        n1, n2 = len(group0), len(group1)
                        s1, s2 = np.std(group0, ddof=1), np.std(group1, ddof=1)
                        
                        # Pooled standard deviation
                        s_pooled = np.sqrt(((n1-1) * s1**2 + (n2-1) * s2**2) / (n1 + n2 - 2))
                        
                        # Cohen's d
                        cohens_d = abs(np.mean(group0) - np.mean(group1)) / s_pooled
                        
                        feature_result['cohens_d'] = float(cohens_d)
                        
                        if cohens_d > 0.8:
                            feature_result['relationship_strength'] = 'strong'
                        elif cohens_d > 0.5:
                            feature_result['relationship_strength'] = 'moderate'
                        elif cohens_d > 0.2:
                            feature_result['relationship_strength'] = 'weak'
                        else:
                            feature_result['relationship_strength'] = 'negligible'
                        
                        # Store feature result
                        results['feature_relationships'][col] = feature_result
                        
                        # Add to feature importance list (only if significant)
                        if p_value < significance_level:
                            feature_importance.append((col, cohens_d, 't_test'))
                
                else:  # Categorical target (not binary)
                    # ANOVA for categorical targets with more than 2 classes
                    category_groups = []
                    
                    for category in valid_data[target_variable].unique():
                        subset = valid_data[valid_data[target_variable] == category][col].values
                        if len(subset) > 0:
                            category_groups.append(subset)
                    
                    if len(category_groups) >= 2 and all(len(g) > 0 for g in category_groups):
                        f_statistic, p_value = stats.f_oneway(*category_groups)
                        
                        # Store results
                        feature_result = {
                            'feature_type': 'numeric',
                            'anova': {
                                'f_statistic': float(f_statistic),
                                'p_value': float(p_value),
                                'significant': p_value < significance_level
                            }
                        }
                        
                        # Calculate effect size (Eta squared)
                        num_categories = len(category_groups)
                        total_samples = sum(len(g) for g in category_groups)
                        
                        eta_squared = (f_statistic * (num_categories - 1)) / (f_statistic * (num_categories - 1) + (total_samples - num_categories))
                        feature_result['eta_squared'] = float(eta_squared)
                        
                        if eta_squared > 0.14:
                            feature_result['relationship_strength'] = 'strong'
                        elif eta_squared > 0.06:
                            feature_result['relationship_strength'] = 'moderate'
                        else:
                            feature_result['relationship_strength'] = 'weak'
                        
                        # Store feature result
                        results['feature_relationships'][col] = feature_result
                        
                        # Add to feature importance list (only if significant)
                        if p_value < significance_level:
                            feature_importance.append((col, eta_squared, 'anova'))
            
            except Exception as e:
                logger.warning(f"Error analyzing relationship between '{col}' and '{target_variable}': {str(e)}")
        
        # Analyze categorical features
        for col in categorical_features:
            try:
                # Skip if too many categories
                if df[col].nunique() > max_categories:
                    continue
                
                # Skip if too many missing values
                if df[col].isna().mean() > 0.5 or df[target_variable].isna().mean() > 0.5:
                    continue
                
                # Get non-missing pairs
                valid_data = df[[col, target_variable]].dropna()
                
                # Skip if too few values
                if len(valid_data) < 10:
                    continue
                
                # Chi-square test for independence
                contingency = pd.crosstab(valid_data[col], valid_data[target_variable])
                
                if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                    chi2, p, dof, expected = stats.chi2_contingency(contingency)
                    
                    # Store results
                    feature_result = {
                        'feature_type': 'categorical',
                        'chi_square': {
                            'statistic': float(chi2),
                            'p_value': float(p),
                            'degrees_of_freedom': int(dof),
                            'significant': p < significance_level
                        }
                    }
                    
                    # Calculate Cramer's V correlation
                    n = contingency.sum().sum()
                    phi2 = chi2 / n
                    r, k = contingency.shape
                    v = np.sqrt(phi2 / min(k-1, r-1))
                    
                    feature_result['cramers_v'] = float(v)
                    
                    # Interpret strength of association
                    if v > 0.5:
                        feature_result['relationship_strength'] = 'strong'
                    elif v > 0.3:
                        feature_result['relationship_strength'] = 'moderate'
                    elif v > 0.1:
                        feature_result['relationship_strength'] = 'weak'
                    else:
                        feature_result['relationship_strength'] = 'negligible'
                    
                    # Store feature result
                    results['feature_relationships'][col] = feature_result
                    
                    # Add to feature importance list (only if significant)
                    if p < significance_level:
                        feature_importance.append((col, v, 'chi_square'))
            
            except Exception as e:
                logger.warning(f"Error analyzing relationship between '{col}' and '{target_variable}': {str(e)}")
    
    # Sort features by importance and get top features
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    top_features = []
    for feature, score, method in feature_importance[:10]:  # Top 10 features
        top_features.append({
            'feature': feature,
            'importance_score': float(score),
            'method': method
        })
    
    results['top_features'] = top_features
    
    # Generate summary
    results['summary'] = {
        'num_features_analyzed': len(results['feature_relationships']),
        'num_significant_features': sum(
            1 for f in results['feature_relationships']
            if results['feature_relationships'][f].get('correlation', {}).get('pearson', {}).get('significant', False) or
            results['feature_relationships'][f].get('correlation', {}).get('spearman', {}).get('significant', False) or
            results['feature_relationships'][f].get('t_test', {}).get('significant', False) or
            results['feature_relationships'][f].get('anova', {}).get('significant', False) or
            results['feature_relationships'][f].get('chi_square', {}).get('significant', False)
        ),
        'num_strong_relationships': sum(
            1 for f in results['feature_relationships']
            if results['feature_relationships'][f].get('relationship_strength') == 'strong'
        )
    }
    
    # Generate recommendations
    # 1. For top features
    if top_features:
        top_feature = top_features[0]['feature']
        
        results['recommendations'].append({
            'severity': 'info',
            'message': f"Feature '{top_feature}' has the strongest relationship with the target variable '{target_variable}'."
        })
    
    # 2. For feature selection
    if results['summary']['num_significant_features'] > 0:
        if results['target_type'] == 'numeric':
            if results['summary']['num_strong_relationships'] > 0:
                strong_features = [f for f in results['feature_relationships'] 
                                if results['feature_relationships'][f].get('relationship_strength') == 'strong']
                
                if strong_features:
                    features_str = ", ".join(strong_features[:3])
                    if len(strong_features) > 3:
                        features_str += f", and {len(strong_features) - 3} more"
                    
                    results['recommendations'].append({
                        'severity': 'medium',
                        'message': f"Consider these features with strong relationships to the target for modeling: {features_str}"
                    })
            else:
                results['recommendations'].append({
                    'severity': 'medium',
                    'message': f"No features have a strong relationship with the target. Consider feature engineering or transformations."
                })
        
        elif results['target_type'] in ['binary', 'categorical']:
            if results['summary']['num_strong_relationships'] > 0:
                strong_features = [f for f in results['feature_relationships'] 
                                if results['feature_relationships'][f].get('relationship_strength') == 'strong']
                
                if strong_features:
                    features_str = ", ".join(strong_features[:3])
                    if len(strong_features) > 3:
                        features_str += f", and {len(strong_features) - 3} more"
                    
                    results['recommendations'].append({
                        'severity': 'medium',
                        'message': f"Consider these features with strong discriminative power for classification: {features_str}"
                    })
    else:
        results['recommendations'].append({
            'severity': 'high',
            'message': "No features show significant relationship with the target variable. Consider feature engineering, transformations, or collecting additional data."
        })
    
    logger.info(f"Target analysis completed. Analyzed {results['summary']['num_features_analyzed']} features, found {results['summary']['num_significant_features']} significant relationships.")
    return results 