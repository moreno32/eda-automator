"""
Analysis functions for unified EDA reports.

This module contains functions for various types of analysis
including data quality, missing data, outliers, and correlations.
"""

import pandas as pd
import numpy as np
from .config import (
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_MISSING_THRESHOLD,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD
)
from .data import get_column_types

def perform_basic_analysis(df):
    """
    Perform basic analysis on the dataset.
    
    Args:
        df (pandas.DataFrame): Data to analyze
    
    Returns:
        dict: Dictionary with analysis results
    """
    results = {}
    
    # Basic info
    results['basic_info'] = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'memory_usage': df.memory_usage().sum() / (1024 * 1024),  # in MB
    }
    
    # Column types
    column_types = get_column_types(df)
    results['column_types'] = column_types
    
    # Number of columns by type
    results['column_counts'] = {
        'numeric': len(column_types['numeric']),
        'categorical': len(column_types['categorical']),
        'datetime': len(column_types['datetime']),
        'text': len(column_types['text']),
        'boolean': len(column_types['boolean']),
        'id': len(column_types['id']),
    }
    
    # Descriptive statistics
    results['descriptive_stats'] = {}
    
    # Numeric columns
    if column_types['numeric']:
        results['descriptive_stats']['numeric'] = df[column_types['numeric']].describe().to_dict()
    
    # Categorical columns
    if column_types['categorical'] or column_types['boolean']:
        cat_cols = column_types['categorical'] + column_types['boolean']
        results['descriptive_stats']['categorical'] = {}
        
        for col in cat_cols:
            value_counts = df[col].value_counts()
            top_categories = value_counts.head(10)
            
            results['descriptive_stats']['categorical'][col] = {
                'unique_values': df[col].nunique(),
                'top_value': value_counts.index[0] if not value_counts.empty else None,
                'top_count': value_counts.iloc[0] if not value_counts.empty else 0,
                'value_counts': dict(zip(top_categories.index.astype(str), top_categories.values)),
            }
    
    # Datetime columns
    if column_types['datetime']:
        results['descriptive_stats']['datetime'] = {}
        
        for col in column_types['datetime']:
            if not df[col].empty:
                results['descriptive_stats']['datetime'][col] = {
                    'min': df[col].min().strftime('%Y-%m-%d') if pd.notna(df[col].min()) else None,
                    'max': df[col].max().strftime('%Y-%m-%d') if pd.notna(df[col].max()) else None,
                    'range_days': (df[col].max() - df[col].min()).days if pd.notna(df[col].min()) and pd.notna(df[col].max()) else None,
                }
    
    return results

def analyze_missing_data(df, threshold=DEFAULT_MISSING_THRESHOLD):
    """
    Analyze missing data in the dataset.
    
    Args:
        df (pandas.DataFrame): Data to analyze
        threshold (float): Threshold for missing data alerts (percentage)
    
    Returns:
        dict: Dictionary with missing data analysis results
    """
    results = {}
    
    # Total missing data
    total_missing = df.isna().sum().sum()
    total_elements = df.shape[0] * df.shape[1]
    total_missing_percentage = 100 * total_missing / total_elements if total_elements > 0 else 0
    
    results['total_missing'] = total_missing
    results['total_elements'] = total_elements
    results['total_missing_percentage'] = total_missing_percentage
    
    # Missing data by column
    column_missing = {}
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_percentage = 100 * missing_count / len(df) if len(df) > 0 else 0
        
        column_missing[col] = {
            'count': int(missing_count),
            'percentage': float(missing_percentage),
            'alert': missing_percentage > threshold
        }
    
    results['column_missing'] = column_missing
    
    # Rows with missing data
    rows_with_missing = df.isna().any(axis=1).sum()
    rows_with_missing_percentage = 100 * rows_with_missing / len(df) if len(df) > 0 else 0
    
    results['rows_with_missing'] = rows_with_missing
    results['rows_with_missing_percentage'] = rows_with_missing_percentage
    
    # Completely missing rows
    completely_missing_rows = df.isna().all(axis=1).sum()
    
    results['completely_missing_rows'] = completely_missing_rows
    
    # Patterns of missingness
    results['missing_patterns'] = _find_missing_patterns(df)
    
    # Alert flag
    results['has_alert'] = total_missing_percentage > threshold
    
    return results

def _find_missing_patterns(df, max_patterns=5):
    """Find common patterns of missing values."""
    # Create a binary matrix of missing values
    missing_binary = df.isna().astype(int)
    
    # Find unique patterns and their counts
    pattern_counts = missing_binary.value_counts().reset_index()
    pattern_counts.columns = list(missing_binary.columns) + ['count']
    
    # Get top patterns
    top_patterns = []
    if len(pattern_counts) > 0:
        top_pattern_rows = pattern_counts.head(max_patterns)
        
        for _, row in top_pattern_rows.iterrows():
            missing_cols = [col for col, val in row.items() if val == 1 and col != 'count']
            
            if missing_cols:
                top_patterns.append({
                    'columns': missing_cols,
                    'count': row['count'],
                    'percentage': 100 * row['count'] / len(df)
                })
            else:
                # This is the pattern with no missing values
                top_patterns.append({
                    'columns': [],
                    'count': row['count'],
                    'percentage': 100 * row['count'] / len(df)
                })
    
    return top_patterns

def analyze_outliers(df):
    """
    Analyze outliers in the dataset.
    
    Args:
        df (pandas.DataFrame): Data to analyze
    
    Returns:
        dict: Dictionary with outliers analysis results
    """
    results = {}
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        results['total_outliers'] = 0
        results['total_outliers_percentage'] = 0
        results['column_outliers'] = {}
        return results
    
    # Calculate outliers for each numeric column using IQR method
    column_outliers = {}
    total_outliers = 0
    
    for col in numeric_cols:
        # Skip columns with all NaN values
        if df[col].isna().all():
            continue
            
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        # Calculate outlier boundaries
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_count = len(outliers)
        outlier_percentage = 100 * outlier_count / df[col].count()
        
        # Add to total count
        total_outliers += outlier_count
        
        # Store result for this column
        column_outliers[col] = {
            'count': outlier_count,
            'percentage': outlier_percentage,
            'lower_threshold': float(lower_bound),
            'upper_threshold': float(upper_bound),
            'min_value': float(df[col].min()),
            'max_value': float(df[col].max()),
            'has_alert': outlier_percentage > DEFAULT_OUTLIER_THRESHOLD * 100,
        }
    
    # Calculate total percentage
    total_numeric_values = df[numeric_cols].count().sum()
    total_outliers_percentage = 100 * total_outliers / total_numeric_values if total_numeric_values > 0 else 0
    
    results['total_outliers'] = total_outliers
    results['total_outliers_percentage'] = total_outliers_percentage
    results['column_outliers'] = column_outliers
    
    return results

def analyze_correlations(df, threshold=DEFAULT_CORRELATION_THRESHOLD):
    """
    Analyze correlations between variables.
    
    Args:
        df (pandas.DataFrame): Data to analyze
        threshold (float): Threshold for correlation alerts
    
    Returns:
        dict: Dictionary with correlation analysis results
    """
    results = {}
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) < 2:
        results['correlation_matrix'] = pd.DataFrame()
        results['strong_correlations'] = []
        results['has_alert'] = False
        return results
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Find strong correlations
    strong_correlations = []
    
    # Get upper triangle of correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            
            if abs(corr) >= threshold:
                strong_correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': float(corr),
                    'is_positive': corr > 0,
                })
    
    # Check for target variable correlations
    target_correlations = {}
    if hasattr(df, 'target') and df.target in numeric_cols:
        target = df.target
        
        for col in numeric_cols:
            if col != target:
                corr = corr_matrix.loc[target, col]
                if abs(corr) >= threshold:
                    target_correlations[col] = float(corr)
    
    # Store results
    results['correlation_matrix'] = corr_matrix
    results['strong_correlations'] = strong_correlations
    results['target_correlations'] = target_correlations if hasattr(df, 'target') else {}
    
    # Alert flag
    results['has_alert'] = len(strong_correlations) > 0
    
    return results

def perform_target_analysis(df, target_variable):
    """
    Perform analysis specific to a target variable.
    
    Args:
        df (pandas.DataFrame): Data to analyze
        target_variable (str): Name of the target variable
    
    Returns:
        dict: Dictionary with target analysis results
    """
    if target_variable not in df.columns:
        return {'error': f"Target variable '{target_variable}' not found in dataset"}
    
    results = {}
    
    # Set target variable in DataFrame
    df.target = target_variable
    results['target_variable'] = target_variable
    
    # Target variable dtype
    target_dtype = df[target_variable].dtype
    results['target_dtype'] = str(target_dtype)
    
    # For numeric target
    if np.issubdtype(target_dtype, np.number):
        results['target_type'] = 'numeric'
        results['target_stats'] = {
            'min': float(df[target_variable].min()),
            'max': float(df[target_variable].max()),
            'mean': float(df[target_variable].mean()),
            'median': float(df[target_variable].median()),
            'std': float(df[target_variable].std()),
            'skew': float(df[target_variable].skew()),
        }
    
    # For categorical target
    elif target_dtype == 'object' or target_dtype.name == 'category':
        results['target_type'] = 'categorical'
        
        # Value counts
        value_counts = df[target_variable].value_counts()
        
        # Class counts
        results['class_counts'] = dict(zip(value_counts.index.astype(str), value_counts.values))
        
        # Class percentages
        class_percentages = 100 * value_counts / len(df)
        results['class_percentages'] = dict(zip(value_counts.index.astype(str), class_percentages.values))
        
        # Check for class imbalance
        class_max = value_counts.max()
        class_min = value_counts.min()
        imbalance_ratio = class_max / class_min if class_min > 0 else float('inf')
        
        results['class_imbalance'] = {
            'imbalance_ratio': float(imbalance_ratio),
            'has_imbalance': imbalance_ratio > 3.0,
            'majority_class': value_counts.index[0],
            'majority_count': int(class_max),
            'minority_class': value_counts.index[-1],
            'minority_count': int(class_min),
        }
    
    # For datetime target
    elif np.issubdtype(target_dtype, np.datetime64):
        results['target_type'] = 'datetime'
        results['target_stats'] = {
            'min': df[target_variable].min().strftime('%Y-%m-%d'),
            'max': df[target_variable].max().strftime('%Y-%m-%d'),
            'range_days': (df[target_variable].max() - df[target_variable].min()).days,
        }
    
    return results

def run_analysis(df, target_variable=None):
    """
    Run a complete analysis on the dataset.
    
    Args:
        df (pandas.DataFrame): Data to analyze
        target_variable (str): Name of the target variable (optional)
    
    Returns:
        EDAAutomator: Instance with analysis results
    """
    from eda_automator import EDAAutomator
    
    print("Initializing EDA analysis...")
    # Create EDA instance
    eda = EDAAutomator(df)
    
    # Validate data
    from .data import validate_data
    df = validate_data(df)
    
    # Run basic analysis
    print("Performing basic analysis...")
    basic_results = perform_basic_analysis(df)
    eda.basic_info = basic_results.get('basic_info', {})
    eda.column_types = basic_results.get('column_types', {})
    eda.column_counts = basic_results.get('column_counts', {})
    eda.descriptive_stats = basic_results.get('descriptive_stats', {})
    
    # Run missing data analysis
    print("Analyzing missing data...")
    missing_results = analyze_missing_data(df)
    eda.missing_data = missing_results
    
    # Run outlier analysis
    print("Analyzing outliers...")
    outlier_results = analyze_outliers(df)
    eda.outliers = outlier_results
    
    # Run correlation analysis
    print("Analyzing correlations...")
    correlation_results = analyze_correlations(df)
    eda.correlations = correlation_results
    
    # Run target analysis if target_variable is provided
    if target_variable is not None and target_variable in df.columns:
        print(f"Performing target analysis for '{target_variable}'...")
        target_results = perform_target_analysis(df, target_variable)
        eda.target_analysis = target_results
    
    # Import and run visualization generators
    from .visualizations import (
        generate_basic_visualizations,
        generate_univariate_plots,
        generate_bivariate_plots
    )
    
    # Generate basic visualizations
    print("Generating basic visualizations...")
    eda.basic_plots = generate_basic_visualizations(df, eda)
    
    # Generate univariate plots
    print("Generating univariate plots...")
    eda.univariate_plots = generate_univariate_plots(df, eda)
    
    # Generate bivariate plots if target is provided
    if target_variable is not None and target_variable in df.columns:
        print("Generating bivariate plots...")
        eda.bivariate_plots = generate_bivariate_plots(df, target_variable, eda)
    
    print("Analysis complete!")
    return eda

def generate_key_findings(eda):
    """
    Generar hallazgos clave basados en el análisis.
    
    Args:
        eda (EDAAutomator): Instancia con los resultados del análisis
    
    Returns:
        str: HTML con los hallazgos clave
    """
    findings = []
    
    # Hallazgos sobre el tamaño del dataset
    if hasattr(eda, 'basic_info'):
        rows = eda.basic_info.get('rows', 0)
        columns = eda.basic_info.get('columns', 0)
        findings.append(f"El dataset tiene {rows:,} filas y {columns} columnas")
    
    # Hallazgos sobre valores faltantes
    if hasattr(eda, 'missing_data'):
        missing_pct = eda.missing_data.get('total_missing_percentage', 0)
        if missing_pct > 0:
            findings.append(f"{missing_pct:.1f}% de los datos están faltantes")
            
            # Columnas con más datos faltantes
            column_missing = eda.missing_data.get('column_missing', {})
            high_missing = [(col, info['percentage']) for col, info in column_missing.items() 
                           if info['percentage'] > 20]
            
            if high_missing:
                high_missing.sort(key=lambda x: x[1], reverse=True)
                if len(high_missing) == 1:
                    findings.append(f"La columna '{high_missing[0][0]}' tiene {high_missing[0][1]:.1f}% de valores faltantes")
                else:
                    cols = ", ".join([f"'{col}' ({pct:.1f}%)" for col, pct in high_missing[:3]])
                    findings.append(f"Las columnas con más valores faltantes son: {cols}")
    
    # Hallazgos sobre outliers
    if hasattr(eda, 'outliers'):
        outliers_pct = eda.outliers.get('total_outliers_percentage', 0)
        if outliers_pct > 0:
            findings.append(f"Se detectaron {outliers_pct:.1f}% de valores atípicos (outliers)")
    
    # Hallazgos sobre correlaciones
    if hasattr(eda, 'correlations'):
        strong_correlations = eda.correlations.get('strong_correlations', [])
        if strong_correlations:
            # Obtener la correlación más fuerte
            strongest = sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)[0]
            var1, var2 = strongest['var1'], strongest['var2']
            corr = strongest['correlation']
            corr_type = "positiva" if corr > 0 else "negativa"
            findings.append(f"La correlación más fuerte ({corr:.2f}) es {corr_type} entre '{var1}' y '{var2}'")
    
    # Hallazgos sobre la variable objetivo
    if hasattr(eda, 'target_analysis'):
        target_var = eda.target_analysis.get('target_variable', '')
        target_type = eda.target_analysis.get('target_type', '')
        
        if target_type == 'categorical':
            class_imbalance = eda.target_analysis.get('class_imbalance', {})
            if class_imbalance.get('has_imbalance', False):
                maj_class = class_imbalance.get('majority_class', '')
                min_class = class_imbalance.get('minority_class', '')
                ratio = class_imbalance.get('imbalance_ratio', 0)
                findings.append(f"La variable objetivo '{target_var}' muestra desbalance de clases ({ratio:.1f}:1) entre '{maj_class}' y '{min_class}'")
        
        elif target_type == 'numeric':
            target_stats = eda.target_analysis.get('target_stats', {})
            if 'mean' in target_stats and 'std' in target_stats:
                findings.append(f"La variable objetivo '{target_var}' tiene media {target_stats['mean']:.2f} y desviación estándar {target_stats['std']:.2f}")
    
    # Unir hallazgos en HTML
    html_findings = ""
    for finding in findings:
        html_findings += f"<li>{finding}</li>\n"
    
    return html_findings

def get_quality_description(score):
    """
    Obtener descripción textual de la calidad de los datos basada en el puntaje.
    
    Args:
        score (float): Puntaje de calidad de los datos (0-10)
    
    Returns:
        str: Descripción textual de la calidad
    """
    if score >= 9.0:
        return "excelente calidad"
    elif score >= 8.0:
        return "muy buena calidad"
    elif score >= 7.0:
        return "buena calidad"
    elif score >= 6.0:
        return "calidad aceptable"
    elif score >= 5.0:
        return "calidad moderada"
    elif score >= 4.0:
        return "calidad cuestionable"
    else:
        return "calidad pobre"

def get_highest_correlation(df):
    """
    Obtener la descripción de la correlación más alta en el dataset.
    
    Args:
        df (pandas.DataFrame): Dataset a analizar
    
    Returns:
        str: Descripción de la correlación más alta
    """
    # Obtener columnas numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) < 2:
        return "No hay suficientes variables numéricas"
    
    # Calcular matriz de correlación
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Obtener correlación más alta (excluyendo la diagonal)
    np.fill_diagonal(corr_matrix.values, 0)
    max_corr = corr_matrix.max().max()
    
    if max_corr < 0.3:
        return "No hay correlaciones significativas"
    
    # Encontrar las variables con la correlación más alta
    max_idx = corr_matrix.stack().idxmax()
    var1, var2 = max_idx
    
    # Determinar si es positiva o negativa
    original_corr = df[numeric_cols].corr().loc[var1, var2]
    corr_type = "+" if original_corr > 0 else "-"
    
    return f"{var1}/{var2} ({corr_type}{max_corr:.2f})" 