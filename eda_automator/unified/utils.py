"""
Módulo de utilidades para los informes EDA unificados.

Este módulo contiene funciones y clases de utilidad para el
procesamiento y generación de informes.
"""

import pandas as pd

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