"""
Distribution analysis functions for EDA Automator

This module provides functions for analyzing the distributions of variables in datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
from scipy import stats

from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def analyze_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    test_normality: bool = True,
    bins: int = 30
) -> Dict[str, Any]:
    """
    Analyze the distributions of variables in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze
    columns : list of str, optional
        Specific columns to analyze. If None, all suitable columns are analyzed
    test_normality : bool, default True
        Whether to test for normality using statistical tests
    bins : int, default 30
        Number of bins to use for histograms and frequency calculations
        
    Returns
    -------
    dict
        Dictionary of distribution analysis results
    """
    logger.info("Analyzing distributions of variables")
    
    # Initialize results
    results = {
        'numeric_distributions': {},
        'categorical_distributions': {},
        'datetime_distributions': {},
        'summary': {},
        'recommendations': []
    }
    
    # Get columns to analyze
    if columns is not None:
        # Validate specified columns
        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            raise ValueError(f"Columns not found in DataFrame: {invalid_columns}")
        analyze_columns = columns
    else:
        analyze_columns = df.columns.tolist()
    
    # Group columns by data type
    numeric_columns = [col for col in analyze_columns 
                      if pd.api.types.is_numeric_dtype(df[col]) and 
                      col in df.columns]
    
    categorical_columns = [col for col in analyze_columns 
                         if (pd.api.types.is_categorical_dtype(df[col]) or 
                            pd.api.types.is_object_dtype(df[col]) or
                            pd.api.types.is_bool_dtype(df[col])) and 
                         col in df.columns]
    
    datetime_columns = [col for col in analyze_columns 
                      if pd.api.types.is_datetime64_dtype(df[col]) and 
                      col in df.columns]
    
    # Analyze numeric distributions
    for col in numeric_columns:
        try:
            # Get non-missing values
            values = df[col].dropna()
            
            # Skip if too few values
            if len(values) < 5:
                continue
                
            # Calculate basic statistics
            stats_dict = {
                'count': len(values),
                'missing_count': df[col].isna().sum(),
                'missing_pct': (df[col].isna().sum() / len(df)) * 100,
                'min': float(values.min()),
                'max': float(values.max()),
                'range': float(values.max() - values.min()),
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'variance': float(values.var()),
                'skewness': float(stats.skew(values)),
                'kurtosis': float(stats.kurtosis(values))
            }
            
            # Calculate percentiles
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                stats_dict[f'percentile_{p}'] = float(np.percentile(values, p))
            
            # Calculate histogram data
            hist, bin_edges = np.histogram(values, bins=bins)
            hist_data = []
            
            for i in range(len(hist)):
                hist_data.append({
                    'bin_start': float(bin_edges[i]),
                    'bin_end': float(bin_edges[i+1]),
                    'count': int(hist[i]),
                    'frequency': float(hist[i] / len(values))
                })
                
            stats_dict['histogram'] = hist_data
            
            # Test for normality if requested
            if test_normality and len(values) > 20:
                # Shapiro-Wilk test (better for smaller samples)
                shapiro_stat, shapiro_p = stats.shapiro(values[:5000])  # Limit to 5000 samples
                
                # D'Agostino's K^2 test
                k2_stat, k2_p = stats.normaltest(values[:5000])  # Limit to 5000 samples
                
                stats_dict['normality_tests'] = {
                    'shapiro': {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    },
                    'dagostino': {
                        'statistic': float(k2_stat),
                        'p_value': float(k2_p),
                        'is_normal': k2_p > 0.05
                    }
                }
                
                # Overall normality assessment
                stats_dict['is_normal'] = shapiro_p > 0.05 or k2_p > 0.05
            
            # Add distribution type assessment
            if abs(stats_dict['skewness']) < 0.5 and abs(stats_dict['kurtosis']) < 0.5:
                stats_dict['distribution_type'] = 'normal'
            elif stats_dict['skewness'] > 1:
                stats_dict['distribution_type'] = 'right_skewed'
            elif stats_dict['skewness'] < -1:
                stats_dict['distribution_type'] = 'left_skewed'
            elif stats_dict['kurtosis'] > 1:
                stats_dict['distribution_type'] = 'heavy_tailed'
            elif stats_dict['kurtosis'] < -1:
                stats_dict['distribution_type'] = 'light_tailed'
            else:
                stats_dict['distribution_type'] = 'approximately_normal'
            
            # Add recommendations for transformations
            if stats_dict['distribution_type'] == 'right_skewed' and stats_dict['min'] >= 0:
                stats_dict['transformation_suggestion'] = 'log'
            elif stats_dict['distribution_type'] == 'left_skewed':
                stats_dict['transformation_suggestion'] = 'square'
            elif abs(stats_dict['skewness']) > 0.5:
                stats_dict['transformation_suggestion'] = 'box-cox'
            else:
                stats_dict['transformation_suggestion'] = None
            
            # Check for modality (uni-modal, bi-modal, multi-modal)
            try:
                from scipy.signal import find_peaks
                
                # Use histogram frequency for peak detection
                hist_values = [item['frequency'] for item in hist_data]
                peaks, _ = find_peaks(hist_values, height=0.05, distance=bins/10)
                
                stats_dict['num_peaks'] = len(peaks)
                
                if len(peaks) == 0:
                    stats_dict['modality'] = 'uniform'
                elif len(peaks) == 1:
                    stats_dict['modality'] = 'unimodal'
                elif len(peaks) == 2:
                    stats_dict['modality'] = 'bimodal'
                else:
                    stats_dict['modality'] = 'multimodal'
            except:
                stats_dict['modality'] = 'unknown'
            
            # Store results
            results['numeric_distributions'][col] = stats_dict
        
        except Exception as e:
            logger.warning(f"Error analyzing distribution for column '{col}': {str(e)}")
    
    # Analyze categorical distributions
    for col in categorical_columns:
        try:
            # Get value counts and frequencies
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = ['value', 'count']
            
            # Calculate statistics
            stats_dict = {
                'count': len(df[col].dropna()),
                'missing_count': df[col].isna().sum(),
                'missing_pct': (df[col].isna().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'mode': df[col].mode()[0] if not df[col].mode().empty else None
            }
            
            # Convert value counts to list of dicts for JSON serialization
            value_list = []
            
            for _, row in value_counts.head(50).iterrows():  # Limit to top 50
                value_list.append({
                    'value': str(row['value']),
                    'count': int(row['count']),
                    'frequency': float(row['count'] / len(df[col].dropna()))
                })
            
            stats_dict['value_counts'] = value_list
            
            # Add Shannon entropy for diversity measurement
            frequencies = np.array([item['frequency'] for item in value_list])
            entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))
            stats_dict['entropy'] = float(entropy)
            
            # Calculate maximum possible entropy for this number of categories
            max_entropy = np.log2(min(len(value_list), len(df[col].dropna())))
            stats_dict['max_entropy'] = float(max_entropy)
            
            # Calculate normalized entropy (0-1)
            if max_entropy > 0:
                stats_dict['normalized_entropy'] = float(entropy / max_entropy)
            else:
                stats_dict['normalized_entropy'] = 0.0
            
            # Assess distribution balance
            if len(value_list) > 0:
                top_freq = value_list[0]['frequency']
                
                if top_freq > 0.9:
                    stats_dict['balance_assessment'] = 'highly_imbalanced'
                elif top_freq > 0.75:
                    stats_dict['balance_assessment'] = 'imbalanced'
                elif top_freq < 0.3 and stats_dict['normalized_entropy'] > 0.7:
                    stats_dict['balance_assessment'] = 'well_balanced'
                else:
                    stats_dict['balance_assessment'] = 'moderately_balanced'
            
            # Store results
            results['categorical_distributions'][col] = stats_dict
        
        except Exception as e:
            logger.warning(f"Error analyzing distribution for column '{col}': {str(e)}")
    
    # Analyze datetime distributions
    for col in datetime_columns:
        try:
            # Get non-missing values
            values = df[col].dropna()
            
            # Skip if too few values
            if len(values) < 5:
                continue
                
            # Calculate basic statistics
            stats_dict = {
                'count': len(values),
                'missing_count': df[col].isna().sum(),
                'missing_pct': (df[col].isna().sum() / len(df)) * 100,
                'min': values.min().strftime('%Y-%m-%d %H:%M:%S'),
                'max': values.max().strftime('%Y-%m-%d %H:%M:%S'),
                'range_days': (values.max() - values.min()).days
            }
            
            # Analyze year distribution
            if hasattr(values, 'dt'):
                year_counts = values.dt.year.value_counts().sort_index()
                year_data = []
                
                for year, count in year_counts.items():
                    year_data.append({
                        'year': int(year),
                        'count': int(count),
                        'frequency': float(count / len(values))
                    })
                
                stats_dict['year_distribution'] = year_data
                
                # Analyze month distribution
                month_counts = values.dt.month.value_counts().sort_index()
                month_data = []
                
                for month, count in month_counts.items():
                    month_data.append({
                        'month': int(month),
                        'count': int(count),
                        'frequency': float(count / len(values))
                    })
                
                stats_dict['month_distribution'] = month_data
                
                # Analyze day of week distribution
                dow_counts = values.dt.dayofweek.value_counts().sort_index()
                dow_data = []
                
                for dow, count in dow_counts.items():
                    dow_data.append({
                        'day_of_week': int(dow),
                        'count': int(count),
                        'frequency': float(count / len(values))
                    })
                
                stats_dict['day_of_week_distribution'] = dow_data
                
                # Analyze hour distribution
                if values.dt.hour.nunique() > 1:
                    hour_counts = values.dt.hour.value_counts().sort_index()
                    hour_data = []
                    
                    for hour, count in hour_counts.items():
                        hour_data.append({
                            'hour': int(hour),
                            'count': int(count),
                            'frequency': float(count / len(values))
                        })
                    
                    stats_dict['hour_distribution'] = hour_data
            
            # Store results
            results['datetime_distributions'][col] = stats_dict
        
        except Exception as e:
            logger.warning(f"Error analyzing distribution for column '{col}': {str(e)}")
    
    # Generate summary
    results['summary'] = {
        'numeric_columns_analyzed': len(results['numeric_distributions']),
        'categorical_columns_analyzed': len(results['categorical_distributions']),
        'datetime_columns_analyzed': len(results['datetime_distributions']),
        'normal_distributions': sum(
            1 for col in results['numeric_distributions'] 
            if results['numeric_distributions'][col].get('is_normal', False)
        ),
        'highly_skewed_distributions': sum(
            1 for col in results['numeric_distributions']
            if abs(results['numeric_distributions'][col].get('skewness', 0)) > 1
        ),
        'highly_imbalanced_categories': sum(
            1 for col in results['categorical_distributions']
            if results['categorical_distributions'][col].get('balance_assessment') == 'highly_imbalanced'
        )
    }
    
    # Generate recommendations
    # 1. For skewed numeric distributions
    highly_skewed = [
        (col, results['numeric_distributions'][col]['skewness'])
        for col in results['numeric_distributions']
        if abs(results['numeric_distributions'][col].get('skewness', 0)) > 1
    ]
    
    if highly_skewed:
        highly_skewed.sort(key=lambda x: abs(x[1]), reverse=True)
        col, skew = highly_skewed[0]
        skew_type = "right" if skew > 0 else "left"
        
        results['recommendations'].append({
            'severity': 'medium',
            'message': f"Column '{col}' has strongly {skew_type}-skewed distribution (skewness={skew:.2f}). Consider applying a transformation.",
            'transformation': results['numeric_distributions'][col].get('transformation_suggestion', 'none')
        })
    
    # 2. For imbalanced categorical variables
    highly_imbalanced = [
        (col, results['categorical_distributions'][col]['value_counts'][0]['frequency'])
        for col in results['categorical_distributions']
        if len(results['categorical_distributions'][col]['value_counts']) > 0 and
        results['categorical_distributions'][col]['balance_assessment'] == 'highly_imbalanced'
    ]
    
    if highly_imbalanced:
        highly_imbalanced.sort(key=lambda x: x[1], reverse=True)
        col, freq = highly_imbalanced[0]
        
        results['recommendations'].append({
            'severity': 'medium',
            'message': f"Column '{col}' is highly imbalanced ({freq*100:.1f}% of values in one category). Consider this for modeling or sampling strategies."
        })
    
    # 3. For multimodal distributions
    multimodal = [
        col for col in results['numeric_distributions']
        if results['numeric_distributions'][col].get('modality') in ['bimodal', 'multimodal']
    ]
    
    if multimodal and len(multimodal) > 0:
        results['recommendations'].append({
            'severity': 'info',
            'message': f"Column '{multimodal[0]}' has a multimodal distribution. Consider if the data might represent multiple populations or clusters."
        })
    
    logger.info(f"Distribution analysis completed for {results['summary']['numeric_columns_analyzed']} numeric, {results['summary']['categorical_columns_analyzed']} categorical, and {results['summary']['datetime_columns_analyzed']} datetime columns.")
    return results

def analyze_pairwise_relationships(
    df: pd.DataFrame,
    x_column: str,
    y_column: str
) -> Dict[str, Any]:
    """
    Analyze the relationship between two variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns
    x_column : str
        Name of the X column
    y_column : str
        Name of the Y column
        
    Returns
    -------
    dict
        Dictionary of relationship analysis results
    """
    logger.info(f"Analyzing relationship between '{x_column}' and '{y_column}'")
    
    # Initialize results
    results = {
        'x_column': x_column,
        'y_column': y_column,
        'relationship_type': None
    }
    
    # Check if columns exist
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in DataFrame")
    
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in DataFrame")
    
    # Get column data types
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x_column])
    y_is_numeric = pd.api.types.is_numeric_dtype(df[y_column])
    
    x_is_categorical = pd.api.types.is_categorical_dtype(df[x_column]) or pd.api.types.is_object_dtype(df[x_column])
    y_is_categorical = pd.api.types.is_categorical_dtype(df[y_column]) or pd.api.types.is_object_dtype(df[y_column])
    
    # Get non-missing pairs
    valid_pairs = df[[x_column, y_column]].dropna()
    
    # Store basic info
    results['valid_pairs'] = len(valid_pairs)
    results['missing_pairs'] = len(df) - len(valid_pairs)
    
    # Case 1: Both numeric
    if x_is_numeric and y_is_numeric:
        results['relationship_type'] = 'numeric_to_numeric'
        
        # Calculate correlation
        pearson_corr, pearson_p = stats.pearsonr(valid_pairs[x_column], valid_pairs[y_column])
        spearman_corr, spearman_p = stats.spearmanr(valid_pairs[x_column], valid_pairs[y_column])
        
        results['correlations'] = {
            'pearson': {
                'coefficient': float(pearson_corr),
                'p_value': float(pearson_p),
                'significant': pearson_p < 0.05
            },
            'spearman': {
                'coefficient': float(spearman_corr),
                'p_value': float(spearman_p),
                'significant': spearman_p < 0.05
            }
        }
        
        # Determine relationship strength
        if abs(pearson_corr) > 0.7 or abs(spearman_corr) > 0.7:
            results['strength'] = 'strong'
        elif abs(pearson_corr) > 0.3 or abs(spearman_corr) > 0.3:
            results['strength'] = 'moderate'
        else:
            results['strength'] = 'weak'
        
        # Determine if linear or non-linear
        abs_diff = abs(abs(pearson_corr) - abs(spearman_corr))
        
        if abs_diff > 0.1:
            # Big difference suggests non-linearity
            results['linearity'] = 'non_linear'
        else:
            results['linearity'] = 'linear'
        
        # Check for potential outlier influence
        x_mean = valid_pairs[x_column].mean()
        x_std = valid_pairs[x_column].std()
        y_mean = valid_pairs[y_column].mean()
        y_std = valid_pairs[y_column].std()
        
        x_outliers = valid_pairs[(valid_pairs[x_column] > x_mean + 3 * x_std) | 
                               (valid_pairs[x_column] < x_mean - 3 * x_std)]
        y_outliers = valid_pairs[(valid_pairs[y_column] > y_mean + 3 * y_std) | 
                               (valid_pairs[y_column] < y_mean - 3 * y_std)]
        
        # If removing outliers significantly changes correlation, flag it
        if len(x_outliers) > 0 or len(y_outliers) > 0:
            clean_data = valid_pairs[(valid_pairs[x_column] <= x_mean + 3 * x_std) & 
                                   (valid_pairs[x_column] >= x_mean - 3 * x_std) &
                                   (valid_pairs[y_column] <= y_mean + 3 * y_std) &
                                   (valid_pairs[y_column] >= y_mean - 3 * y_std)]
            
            if len(clean_data) > 10:  # Need enough points for correlation
                clean_pearson, _ = stats.pearsonr(clean_data[x_column], clean_data[y_column])
                pearson_diff = abs(pearson_corr - clean_pearson)
                
                results['outlier_influence'] = {
                    'outlier_count': len(x_outliers) + len(y_outliers),
                    'correlation_change': float(pearson_diff),
                    'significant_change': pearson_diff > 0.1
                }
    
    # Case 2: Numeric to categorical
    elif (x_is_numeric and y_is_categorical) or (x_is_categorical and y_is_numeric):
        results['relationship_type'] = 'numeric_to_categorical'
        
        # Set numeric and categorical columns
        num_col = y_column if y_is_numeric else x_column
        cat_col = x_column if y_is_numeric else y_column
        
        # Calculate stats by category
        category_stats = {}
        
        for category in df[cat_col].dropna().unique():
            subset = df[df[cat_col] == category][num_col].dropna()
            
            if len(subset) > 0:
                category_stats[str(category)] = {
                    'count': int(len(subset)),
                    'mean': float(subset.mean()),
                    'median': float(subset.median()),
                    'std': float(subset.std()) if len(subset) > 1 else 0.0,
                    'min': float(subset.min()),
                    'max': float(subset.max())
                }
        
        results['category_stats'] = category_stats
        
        # Perform ANOVA to test for significant differences
        try:
            category_groups = []
            category_names = []
            
            for category in df[cat_col].dropna().unique():
                values = df[df[cat_col] == category][num_col].dropna().values
                if len(values) > 0:
                    category_groups.append(values)
                    category_names.append(str(category))
            
            if len(category_groups) >= 2 and all(len(g) > 0 for g in category_groups):
                f_statistic, p_value = stats.f_oneway(*category_groups)
                
                results['anova'] = {
                    'f_statistic': float(f_statistic),
                    'p_value': float(p_value),
                    'significant_difference': p_value < 0.05
                }
        except Exception as e:
            logger.warning(f"Error performing ANOVA: {str(e)}")
    
    # Case 3: Both categorical
    elif x_is_categorical and y_is_categorical:
        results['relationship_type'] = 'categorical_to_categorical'
        
        # Create contingency table
        try:
            contingency = pd.crosstab(valid_pairs[x_column], valid_pairs[y_column])
            
            # Convert to nested list for JSON serialization
            crosstab = []
            
            for idx, row in enumerate(contingency.index):
                row_data = {'category': str(row)}
                
                for col in contingency.columns:
                    row_data[str(col)] = int(contingency.iloc[idx][col])
                
                crosstab.append(row_data)
            
            results['contingency_table'] = crosstab
            
            # Chi-square test for independence
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            results['chi_square'] = {
                'statistic': float(chi2),
                'p_value': float(p),
                'degrees_of_freedom': int(dof),
                'significant': p < 0.05
            }
            
            # Calculate Cramer's V correlation
            n = contingency.sum().sum()
            phi2 = chi2 / n
            r, k = contingency.shape
            v = np.sqrt(phi2 / min(k-1, r-1))
            
            results['cramers_v'] = float(v)
            
            # Interpret strength of association
            if v < 0.1:
                results['association_strength'] = 'negligible'
            elif v < 0.2:
                results['association_strength'] = 'weak'
            elif v < 0.4:
                results['association_strength'] = 'moderate'
            elif v < 0.6:
                results['association_strength'] = 'relatively_strong'
            elif v < 0.8:
                results['association_strength'] = 'strong'
            else:
                results['association_strength'] = 'very_strong'
        
        except Exception as e:
            logger.warning(f"Error analyzing categorical relationship: {str(e)}")
    
    # Generate recommendations
    results['recommendations'] = []
    
    if results['relationship_type'] == 'numeric_to_numeric':
        # Recommend based on correlation strength
        corr = results['correlations']['pearson']['coefficient']
        
        if abs(corr) > 0.9:
            results['recommendations'].append({
                'severity': 'high',
                'message': f"Very strong correlation ({corr:.2f}) between '{x_column}' and '{y_column}'. Consider removing one of these variables to reduce multicollinearity."
            })
        elif abs(corr) > 0.7:
            results['recommendations'].append({
                'severity': 'medium',
                'message': f"Strong correlation ({corr:.2f}) between '{x_column}' and '{y_column}'. Be cautious about multicollinearity in regression models."
            })
        
        # Recommend based on linearity
        if results.get('linearity') == 'non_linear' and abs(corr) > 0.3:
            results['recommendations'].append({
                'severity': 'info',
                'message': f"Non-linear relationship detected between '{x_column}' and '{y_column}'. Consider non-linear transformations or models."
            })
        
        # Recommend based on outlier influence
        if results.get('outlier_influence', {}).get('significant_change', False):
            results['recommendations'].append({
                'severity': 'medium',
                'message': f"Correlation between '{x_column}' and '{y_column}' is significantly influenced by outliers. Consider robust correlation measures or outlier handling."
            })
    
    elif results['relationship_type'] == 'numeric_to_categorical':
        # Recommend based on ANOVA
        if results.get('anova', {}).get('significant_difference', False):
            results['recommendations'].append({
                'severity': 'info',
                'message': f"Significant differences in '{num_col}' values across '{cat_col}' categories. This feature may be informative for predictive modeling."
            })
    
    elif results['relationship_type'] == 'categorical_to_categorical':
        # Recommend based on Cramer's V
        if results.get('cramers_v', 0) > 0.5:
            results['recommendations'].append({
                'severity': 'medium',
                'message': f"Strong association between '{x_column}' and '{y_column}' (Cramer's V = {results['cramers_v']:.2f}). Consider consolidating these variables or addressing potential data leakage."
            })
    
    logger.info(f"Relationship analysis completed between '{x_column}' and '{y_column}'")
    return results 