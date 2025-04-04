"""
Generador de informes en formato Markdown.

Este módulo contiene funciones para generar informes Markdown
a partir de los resultados del análisis EDA.
"""

import os
import pandas as pd
import datetime
from ..analysis import get_quality_description, get_highest_correlation

def generate_markdown_report(output_path, eda):
    """
    Genera un informe en formato Markdown a partir de los resultados del análisis.
    
    Args:
        output_path (str): Ruta donde se guardará el informe Markdown
        eda (EDAAutomator): Instancia de EDA con los resultados
        
    Returns:
        str: Ruta al archivo Markdown generado
    """
    print("\nGenerando informe Markdown...")
    
    # Asegurar que el directorio de salida exista
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener el DataFrame y resultados
    df = eda.df
    results = eda.results
    
    # Obtener info básica
    dataset_name = getattr(df, 'name', 'Dataset')
    num_rows, num_cols = df.shape
    target_variable = results.get('target_variable', 'N/A')
    
    # Métricas clave
    data_quality_score = results.get('data_quality', {}).get('overall_score', 0)
    quality_description = get_quality_description(data_quality_score)
    missing_pct = results.get('missing_data', {}).get('total_missing_percentage', 0)
    outliers_pct = results.get('outliers', {}).get('total_outliers_percentage', 0)
    highest_corr = get_highest_correlation(df)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Título y encabezado
        f.write(f"# Informe de Análisis Exploratorio de Datos\n\n")
        f.write(f"**{dataset_name}** | {num_rows} filas, {num_cols} columnas | " + 
                f"Generado: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
        
        # Resumen Ejecutivo
        f.write("## 📊 Resumen Ejecutivo\n\n")
        
        # Tabla de métricas
        f.write("| Métrica | Valor | Descripción |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| **Calidad de Datos** | **{data_quality_score:.1f}/10** | {quality_description.title()} |\n")
        f.write(f"| **Valores Faltantes** | {missing_pct:.1f}% | del total de datos |\n")
        f.write(f"| **Outliers** | {outliers_pct:.1f}% | de valores atípicos |\n")
        f.write(f"| **Mayor Correlación** | {highest_corr} | entre variables |\n\n")
        
        # Hallazgos Clave
        f.write("### 🔍 Hallazgos Clave\n\n")
        
        # Extraer hallazgos del HTML y convertirlos a Markdown
        findings_html = results.get('key_findings_html', '')
        if findings_html:
            # Convertir elementos de lista HTML a Markdown
            findings = findings_html.replace('<li>', '- ').replace('</li>', '\n')
            f.write(findings + "\n\n")
        else:
            # Generar hallazgos en formato Markdown
            findings = []
            
            # Hallazgo 1: Calidad de datos
            if data_quality_score > 0:
                findings.append(f"- La calidad general de los datos es {quality_description} ({data_quality_score:.1f}/10).")
            
            # Hallazgo 2: Datos faltantes
            if missing_pct > 5:
                findings.append(f"- Datos incompletos: {missing_pct:.1f}% de valores faltantes en total.")
            
            # Hallazgo 3: Correlaciones
            if 'correlations' in results and 'high_correlations' in results['correlations']:
                high_corrs = results['correlations']['high_correlations']
                if high_corrs and len(high_corrs) > 0:
                    top_corr = high_corrs[0]
                    findings.append(f"- Alta correlación detectada entre {top_corr[0]} y {top_corr[1]} ({top_corr[2]:.2f}).")
            
            # Escribir hallazgos
            f.write("\n".join(findings) + "\n\n")
        
        # Alertas y Problemas
        f.write("### ⚠️ Alertas y Problemas Detectados\n\n")
        
        # Datos faltantes por variable
        missing_data = results.get('missing_data', {})
        if 'missing_by_variable' in missing_data and missing_data['missing_by_variable']:
            high_missing_vars = []
            for var, info in missing_data['missing_by_variable'].items():
                if info.get('percentage', 0) > 20:  # Variables con más de 20% de valores faltantes
                    high_missing_vars.append((var, info.get('percentage', 0)))
            
            if high_missing_vars:
                f.write("**Variables con muchos datos faltantes:**\n\n")
                for var, pct in sorted(high_missing_vars, key=lambda x: x[1], reverse=True):
                    f.write(f"- **{var}**: {pct:.1f}% valores faltantes\n")
                f.write("\n")
        
        # Outliers por variable
        outliers = results.get('outliers', {})
        if 'variables_with_outliers' in outliers and outliers['variables_with_outliers']:
            vars_with_outliers = []
            for var, info in outliers['variables_with_outliers'].items():
                if info.get('percentage', 0) > 5:  # Variables con más de 5% de outliers
                    vars_with_outliers.append((var, info.get('count', 0), info.get('percentage', 0)))
            
            if vars_with_outliers:
                f.write("**Variables con muchos outliers:**\n\n")
                for var, count, pct in sorted(vars_with_outliers, key=lambda x: x[2], reverse=True):
                    f.write(f"- **{var}**: {count} outliers ({pct:.1f}%)\n")
                f.write("\n")
        
        # Estadísticas Descriptivas
        f.write("## 📈 Estadísticas Descriptivas\n\n")
        
        # Convertir describe() a Markdown
        desc_df = df.describe().round(2).T
        desc_md = desc_df.to_markdown()
        f.write(desc_md + "\n\n")
        
        # Variables Importantes
        f.write("## 🔑 Variables Importantes\n\n")
        
        # Correlaciones con target (si hay)
        if target_variable != 'N/A' and 'correlations' in results and 'target_correlations' in results['correlations']:
            target_corrs = results['correlations']['target_correlations']
            if target_corrs:
                f.write(f"### Correlaciones con {target_variable}\n\n")
                f.write("| Variable | Correlación |\n")
                f.write("| --- | --- |\n")
                
                # Ordenar por valor absoluto de correlación, mostrar solo top 10
                sorted_corrs = sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                for var, corr in sorted_corrs:
                    f.write(f"| {var} | {corr:.3f} |\n")
                f.write("\n")
        
        # Distribución de variables categóricas (si las hay)
        if 'categorical_variables' in results.get('data_types', {}):
            cat_vars = results['data_types']['categorical_variables']
            if cat_vars and len(cat_vars) > 0:
                f.write("### Distribución de Variables Categóricas\n\n")
                
                for var in cat_vars[:5]:  # Limitar a 5 variables para no hacer el informe demasiado largo
                    if var in df.columns:
                        f.write(f"#### {var}\n\n")
                        value_counts = df[var].value_counts()
                        
                        # Convertir a Markdown
                        f.write("| Valor | Frecuencia | Porcentaje |\n")
                        f.write("| --- | --- | --- |\n")
                        
                        for val, count in value_counts.items():
                            pct = 100 * count / len(df)
                            f.write(f"| {val} | {count} | {pct:.1f}% |\n")
                        
                        f.write("\n")
        
        # Recomendaciones
        f.write("## 💡 Recomendaciones\n\n")
        
        # Extraer recomendaciones del HTML y convertirlas a Markdown
        recommendations_html = results.get('recommendations_html', '')
        if recommendations_html:
            # Convertir elementos de lista HTML a Markdown
            recommendations = recommendations_html.replace('<li>', '- ').replace('</li>', '\n')
            f.write(recommendations + "\n\n")
        else:
            # Recomendaciones genéricas
            f.write("- Explore relaciones adicionales entre variables mediante gráficos bivariados.\n")
            f.write("- Considere técnicas de selección de características para modelos predictivos.\n")
            f.write("- Evalúe diferentes algoritmos de aprendizaje automático para su problema específico.\n\n")
        
        # Pie de página
        f.write("---\n\n")
        f.write("*Este informe fue generado automáticamente con [EDA Automator](https://github.com/moreno32/eda-automator)*")
    
    print(f"Informe Markdown guardado en {output_path}")
    return output_path 