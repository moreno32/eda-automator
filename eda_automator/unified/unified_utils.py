"""
Utility functions for the EDA Automator package.

This module contains various utility functions used across the EDA Automator package.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

def setup_logging(log_file=None, level=logging.INFO):
    """
    Set up logging for the EDA Automator package.
    
    Args:
        log_file (str, optional): Path to log file. If None, logs to console.
        level (int, optional): Logging level. Defaults to INFO.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('eda_automator')
    logger.setLevel(level)
    
    # Remove existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def infer_data_types(df):
    """
    Infer the data types of DataFrame columns.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary with column types (numeric, categorical, datetime, etc.)
    """
    column_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'boolean': [],
        'text': [],
        'id': []
    }
    
    for col in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's likely an ID column
            if is_likely_id(df[col]):
                column_types['id'].append(col)
            # Check if it's boolean-like
            elif set(df[col].dropna().unique()).issubset({0, 1}):
                column_types['boolean'].append(col)
            else:
                column_types['numeric'].append(col)
        # Check if column is datetime
        elif pd.api.types.is_datetime64_dtype(df[col]) or is_datetime_like(df[col]):
            column_types['datetime'].append(col)
        # Check if column is boolean
        elif pd.api.types.is_bool_dtype(df[col]):
            column_types['boolean'].append(col)
        # Check if it's likely a text field
        elif is_likely_text(df[col]):
            column_types['text'].append(col)
        # Otherwise, consider it categorical
        else:
            column_types['categorical'].append(col)
    
    return column_types

def is_likely_id(series):
    """
    Check if a series is likely to be an ID column.
    
    Args:
        series (pandas.Series): Input series
    
    Returns:
        bool: True if the series is likely an ID column
    """
    # If it has many unique values relative to its length
    if series.nunique() / len(series) > 0.8:
        # And it's numeric with mostly unique values
        if pd.api.types.is_numeric_dtype(series):
            return True
        # Or it's a string with patterns like IDs (contains numbers)
        elif pd.api.types.is_string_dtype(series):
            # Check if strings match common ID patterns
            has_ids = series.dropna().astype(str).str.match(r'.*\d+.*').mean() > 0.8
            return has_ids
    
    return False

def is_datetime_like(series):
    """
    Check if a series contains datetime-like strings.
    
    Args:
        series (pandas.Series): Input series
    
    Returns:
        bool: True if the series likely contains datetime values
    """
    if pd.api.types.is_string_dtype(series):
        # Try to convert to datetime
        try:
            pd.to_datetime(series, errors='raise')
            return True
        except (ValueError, TypeError):
            # Try with a sample for performance
            sample = series.dropna().sample(min(100, len(series)), random_state=42)
            try:
                pd.to_datetime(sample, errors='raise')
                return True
            except (ValueError, TypeError):
                return False
    
    return False

def is_likely_text(series):
    """
    Check if a series likely contains text data (not categorical).
    
    Args:
        series (pandas.Series): Input series
    
    Returns:
        bool: True if the series likely contains text data
    """
    if not pd.api.types.is_string_dtype(series):
        return False
    
    # If too many unique values or long strings, it's likely text
    n_unique = series.nunique()
    if n_unique > 100 or n_unique / len(series) > 0.5:
        # Check average string length
        avg_length = series.dropna().astype(str).str.len().mean()
        return avg_length > 20
    
    return False

def detect_data_issues(df):
    """
    Detect common data issues in a DataFrame.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary with data issues
    """
    issues = {
        'empty_dataframe': df.empty,
        'duplicate_rows': df.duplicated().sum(),
        'missing_values': df.isnull().sum().sum(),
        'missing_values_pct': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'columns_with_missing': [col for col in df.columns if df[col].isnull().any()],
        'constant_columns': [col for col in df.columns if df[col].nunique() <= 1],
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    }
    
    return issues

def create_timestamp_id():
    """
    Create a timestamp-based ID for reports.
    
    Returns:
        str: Timestamp-based ID
    """
    return datetime.now().strftime('%Y%m%d%H%M%S')

def format_number(value, precision=2):
    """
    Format a number with commas and specified precision.
    
    Args:
        value (float): Number to format
        precision (int, optional): Decimal precision. Defaults to 2.
    
    Returns:
        str: Formatted number string
    """
    if pd.isna(value):
        return "N/A"
    
    if isinstance(value, int) or value.is_integer():
        return f"{int(value):,}"
    
    return f"{value:,.{precision}f}"

def summarize_dataframe(df):
    """
    Create a comprehensive summary of a DataFrame.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
    
    Returns:
        dict: Summary statistics and information
    """
    if df.empty:
        return {
            'rows': 0,
            'columns': 0,
            'memory_usage': 0,
            'column_stats': {}
        }
    
    # Basic info
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        'column_stats': {}
    }
    
    # Column statistics
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique()
        }
        
        # Add type-specific statistics
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                'min': df[col].min() if not df[col].isnull().all() else None,
                'max': df[col].max() if not df[col].isnull().all() else None,
                'mean': df[col].mean() if not df[col].isnull().all() else None,
                'median': df[col].median() if not df[col].isnull().all() else None,
                'std': df[col].std() if not df[col].isnull().all() else None
            })
        elif pd.api.types.is_string_dtype(df[col]):
            # Calculate value counts for top categories
            if df[col].nunique() < 20:
                col_stats['value_counts'] = df[col].value_counts().to_dict()
            
            # Calculate average string length
            col_stats['avg_length'] = df[col].dropna().astype(str).str.len().mean()
        
        summary['column_stats'][col] = col_stats
    
    return summary

def safe_divide(a, b):
    """
    Safely divide two numbers, returning 0 if the denominator is 0.
    
    Args:
        a (float): Numerator
        b (float): Denominator
    
    Returns:
        float: Result of division or 0 if division by zero
    """
    return a / b if b != 0 else 0

def generate_alerts_html(eda):
    """
    Genera alertas HTML basadas en los resultados del análisis.
    
    Args:
        eda (EDAAutomator): Instancia con los resultados del análisis
    
    Returns:
        str: Código HTML con las alertas
    """
    alerts_html = ""
    
    # Alerta de datos faltantes
    if hasattr(eda, 'missing_data'):
        missing_data = eda.missing_data
        total_missing_pct = missing_data.get('total_missing_percentage', 0)
        
        if total_missing_pct > 10:
            alerts_html += f"""
            <div class="alert alert-danger">
                <div class="alert-content">
                    <h4>Datos Faltantes - Nivel Alto ({total_missing_pct:.1f}%)</h4>
                    <p>El dataset contiene un porcentaje elevado de valores faltantes. 
                    Se recomienda utilizar técnicas de imputación o considerar eliminar las variables más afectadas.</p>
                </div>
            </div>
            """
        elif total_missing_pct > 5:
            alerts_html += f"""
            <div class="alert alert-warning">
                <div class="alert-content">
                    <h4>Datos Faltantes - Nivel Medio ({total_missing_pct:.1f}%)</h4>
                    <p>El dataset contiene algunos valores faltantes. 
                    Considere utilizar técnicas de imputación para las variables afectadas.</p>
                </div>
            </div>
            """
        elif total_missing_pct > 0:
            alerts_html += f"""
            <div class="alert alert-success">
                <div class="alert-content">
                    <h4>Datos Faltantes - Nivel Bajo ({total_missing_pct:.1f}%)</h4>
                    <p>El dataset contiene un porcentaje bajo de valores faltantes. 
                    Estos pueden ser manejados fácilmente mediante técnicas de imputación.</p>
                </div>
            </div>
            """
    
    # Alerta de outliers
    if hasattr(eda, 'outliers'):
        outliers = eda.outliers
        outliers_pct = outliers.get('total_outliers_percentage', 0)
        
        if outliers_pct > 10:
            alerts_html += f"""
            <div class="alert alert-danger">
                <div class="alert-content">
                    <h4>Valores Atípicos - Nivel Alto ({outliers_pct:.1f}%)</h4>
                    <p>El dataset contiene una gran cantidad de valores atípicos que pueden afectar los análisis.
                    Considere transformar las variables o usar técnicas robustas a outliers.</p>
                </div>
            </div>
            """
        elif outliers_pct > 5:
            alerts_html += f"""
            <div class="alert alert-warning">
                <div class="alert-content">
                    <h4>Valores Atípicos - Nivel Medio ({outliers_pct:.1f}%)</h4>
                    <p>El dataset contiene algunos valores atípicos que podrían afectar ciertos análisis.
                    Considere revisar estos valores antes de realizar modelos sensibles a outliers.</p>
                </div>
            </div>
            """
    
    # Alerta de correlaciones
    if hasattr(eda, 'correlations'):
        correlations = eda.correlations
        strong_correlations = correlations.get('strong_correlations', [])
        
        if len(strong_correlations) > 5:
            alerts_html += f"""
            <div class="alert alert-warning">
                <div class="alert-content">
                    <h4>Múltiples Correlaciones Fuertes Detectadas</h4>
                    <p>El dataset contiene muchas variables numéricas altamente correlacionadas.
                    Esto podría causar problemas de multicolinealidad en modelos predictivos.</p>
                </div>
            </div>
            """
        elif len(strong_correlations) > 0:
            # Tomar la correlación más fuerte
            strongest = sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)[0]
            var1 = strongest['var1']
            var2 = strongest['var2']
            corr_val = strongest['correlation']
            corr_type = "positiva" if corr_val > 0 else "negativa"
            
            alerts_html += f"""
            <div class="alert alert-success">
                <div class="alert-content">
                    <h4>Correlación Fuerte Detectada</h4>
                    <p>Se encontró una correlación {corr_type} fuerte ({corr_val:.2f}) entre las variables
                    '{var1}' y '{var2}'.</p>
                </div>
            </div>
            """
    
    # Si no hay alertas, mostrar mensaje positivo
    if not alerts_html:
        alerts_html = """
        <div class="alert alert-success">
            <div class="alert-content">
                <h4>¡Datos en Buen Estado!</h4>
                <p>No se detectaron problemas significativos en los datos. Todos los indicadores de calidad están dentro de rangos aceptables.</p>
            </div>
        </div>
        """
    
    return alerts_html

def generate_recommendations(eda):
    """
    Genera recomendaciones HTML basadas en los resultados del análisis.
    
    Args:
        eda (EDAAutomator): Instancia con los resultados del análisis
    
    Returns:
        str: Código HTML con las recomendaciones
    """
    recommendations = []
    
    # Recomendaciones para datos faltantes
    if hasattr(eda, 'missing_data'):
        missing_data = eda.missing_data
        column_missing = missing_data.get('column_missing', {})
        
        # Columnas con más del 20% de datos faltantes
        high_missing_cols = [col for col, info in column_missing.items() if info.get('percentage', 0) > 20]
        
        if high_missing_cols:
            cols_str = ", ".join([f"'{col}'" for col in high_missing_cols[:3]])
            if len(high_missing_cols) > 3:
                cols_str += f" y {len(high_missing_cols) - 3} más"
            
            recommendations.append(f"""
            <li>Considere eliminar o imputar las columnas con alto porcentaje de valores faltantes: {cols_str}.</li>
            """)
    
    # Recomendaciones para outliers
    if hasattr(eda, 'outliers'):
        outliers = eda.outliers
        column_outliers = outliers.get('column_outliers', {})
        
        # Columnas con más del 10% de outliers
        high_outlier_cols = [col for col, info in column_outliers.items() if info.get('percentage', 0) > 10]
        
        if high_outlier_cols:
            cols_str = ", ".join([f"'{col}'" for col in high_outlier_cols[:3]])
            if len(high_outlier_cols) > 3:
                cols_str += f" y {len(high_outlier_cols) - 3} más"
            
            recommendations.append(f"""
            <li>Aplique transformaciones (logarítmica, raíz cuadrada) a variables con muchos outliers: {cols_str}.</li>
            """)
    
    # Recomendaciones para correlaciones fuertes
    if hasattr(eda, 'correlations'):
        correlations = eda.correlations
        strong_correlations = correlations.get('strong_correlations', [])
        
        if len(strong_correlations) > 5:
            recommendations.append("""
            <li>Utilice técnicas de reducción de dimensionalidad como PCA para manejar las múltiples variables correlacionadas.</li>
            """)
    
    # Recomendaciones para target variable
    if hasattr(eda, 'target_analysis'):
        target_analysis = eda.target_analysis
        if target_analysis.get('target_type') == 'categorical':
            class_imbalance = target_analysis.get('class_imbalance', {})
            if class_imbalance.get('has_imbalance', False):
                recommendations.append("""
                <li>La variable objetivo muestra un desbalance de clases. Considere utilizar técnicas como SMOTE, 
                subsampling/oversampling o ajustar los pesos de las clases en el modelo.</li>
                """)
    
    # Recomendaciones generales
    recommendations.append("""
    <li>Normalice las variables numéricas para mejorar el rendimiento de algoritmos sensibles a la escala (como k-means, SVM, redes neuronales).</li>
    """)
    
    recommendations.append("""
    <li>Considere realizar feature engineering para crear nuevas variables que puedan capturar relaciones no lineales.</li>
    """)
    
    # Convertir lista a HTML
    recommendations_html = "<ul>\n" + "\n".join(recommendations) + "\n</ul>"
    
    return recommendations_html

def generate_key_findings(eda):
    """
    Genera hallazgos clave basados en los resultados del análisis.
    
    Args:
        eda (EDAAutomator): Instancia con los resultados del análisis
    
    Returns:
        str: Código HTML con los hallazgos clave
    """
    findings = []
    
    # Dataset y tamaño
    df = eda.df
    num_rows, num_cols = df.shape
    findings.append(f"""
    <li>El dataset contiene <strong>{num_rows:,}</strong> filas y <strong>{num_cols}</strong> columnas.</li>
    """)
    
    # Distribución de tipos de datos
    if hasattr(eda, 'column_counts'):
        column_counts = eda.column_counts
        numeric_count = column_counts.get('numeric', 0)
        cat_count = column_counts.get('categorical', 0) + column_counts.get('boolean', 0)
        date_count = column_counts.get('datetime', 0)
        
        findings.append(f"""
        <li>Composición del dataset: <strong>{numeric_count}</strong> variables numéricas, 
        <strong>{cat_count}</strong> categóricas y <strong>{date_count}</strong> de fecha/hora.</li>
        """)
    
    # Datos faltantes
    if hasattr(eda, 'missing_data'):
        missing_data = eda.missing_data
        total_missing_pct = missing_data.get('total_missing_percentage', 0)
        
        findings.append(f"""
        <li>El <strong>{total_missing_pct:.1f}%</strong> de los datos están faltantes en todo el dataset.</li>
        """)
    
    # Outliers
    if hasattr(eda, 'outliers'):
        outliers = eda.outliers
        total_outliers_pct = outliers.get('total_outliers_percentage', 0)
        
        findings.append(f"""
        <li>Se detectaron <strong>{total_outliers_pct:.1f}%</strong> de valores atípicos (outliers) en las variables numéricas.</li>
        """)
    
    # Correlaciones fuertes
    if hasattr(eda, 'correlations'):
        correlations = eda.correlations
        strong_correlations = correlations.get('strong_correlations', [])
        
        if strong_correlations:
            strongest = sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)[0]
            var1 = strongest['var1']
            var2 = strongest['var2']
            corr_val = strongest['correlation']
            corr_type = "positiva" if corr_val > 0 else "negativa"
            
            findings.append(f"""
            <li>La correlación más fuerte es entre '{var1}' y '{var2}', con una correlación {corr_type} de <strong>{corr_val:.2f}</strong>.</li>
            """)
    
    # Variable objetivo
    if hasattr(eda, 'target_analysis'):
        target_analysis = eda.target_analysis
        target_variable = target_analysis.get('target_variable', 'N/A')
        target_type = target_analysis.get('target_type', 'N/A')
        
        if target_type == 'categorical':
            class_counts = target_analysis.get('class_counts', {})
            if class_counts:
                classes_str = ", ".join([f"'{k}': {v}" for k, v in list(class_counts.items())[:3]])
                findings.append(f"""
                <li>La variable objetivo '{target_variable}' es categórica con la siguiente distribución: {classes_str}.</li>
                """)
        elif target_type == 'numeric':
            target_stats = target_analysis.get('target_stats', {})
            if 'mean' in target_stats and 'std' in target_stats:
                findings.append(f"""
                <li>La variable objetivo '{target_variable}' es numérica con media <strong>{target_stats['mean']:.2f}</strong> 
                y desviación estándar <strong>{target_stats['std']:.2f}</strong>.</li>
                """)
    
    # Convertir lista a HTML
    findings_html = "<ul>\n" + "\n".join(findings) + "\n</ul>"
    
    return findings_html 