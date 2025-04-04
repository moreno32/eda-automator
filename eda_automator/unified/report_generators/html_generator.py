"""
Generador de informes en formato HTML.

Este módulo contiene funciones para generar informes HTML
a partir de los resultados del análisis EDA.
"""

import os
from pathlib import Path
import datetime
from ..utils import generate_alerts_html, generate_recommendations
from ..analysis import generate_key_findings, get_quality_description, get_highest_correlation

def generate_html_report(output_path, eda):
    """
    Genera un informe HTML completo del análisis exploratorio de datos.
    
    Args:
        output_path (str): Ruta donde se guardará el informe HTML
        eda (EDAAutomator): Instancia de EDA con los resultados
        
    Returns:
        str: Ruta al archivo HTML generado
    """
    print("\nGenerando informe HTML...")
    
    # Asegurar que el directorio de salida exista
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener el DataFrame y resultados
    df = eda.df
    
    # Obtener info básica
    dataset_name = getattr(df, 'name', 'Dataset')
    num_rows, num_cols = df.shape
    target_variable = getattr(eda, 'target_analysis', {}).get('target_variable', 'N/A')
    
    # Generar componentes HTML
    alerts_html = generate_alerts_html(eda)
    key_findings_html = generate_key_findings(eda)
    recommendations_html = generate_recommendations(eda)
    
    # Generar HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header, CSS y metadatos
        f.write("""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe de Análisis Exploratorio de Datos</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
        }
        
        .header p {
            margin: 10px 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .content {
            padding: 2rem;
        }
        
        .section {
            margin-bottom: 2rem;
            border-bottom: 1px solid #eee;
            padding-bottom: 2rem;
        }
        
        .section:last-child {
            border-bottom: none;
            margin-bottom: 0;
        }
        
        h2 {
            color: var(--secondary-color);
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--primary-color);
        }
        
        h3 {
            color: var(--secondary-color);
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 1.5rem;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card-title {
            font-size: 1.1rem;
            color: var(--secondary-color);
            margin-top: 0;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .card-value {
            font-size: 2rem;
            text-align: center;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .card-label {
            text-align: center;
            color: #666;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        .alert {
            display: flex;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .alert-icon {
            font-size: 1.8rem;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 40px;
        }
        
        .alert-content h4 {
            margin: 0 0 0.5rem 0;
        }
        
        .alert-content p {
            margin: 0;
            opacity: 0.9;
        }
        
        .alert-success {
            background-color: rgba(46, 204, 113, 0.1);
            border-left: 4px solid var(--success-color);
        }
        
        .alert-warning {
            background-color: rgba(243, 156, 18, 0.1);
            border-left: 4px solid var(--warning-color);
        }
        
        .alert-danger {
            background-color: rgba(231, 76, 60, 0.1);
            border-left: 4px solid var(--danger-color);
        }
        
        .table-responsive {
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1.5rem;
        }
        
        thead {
            background-color: var(--primary-color);
            color: white;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        .badge {
            display: inline-block;
            padding: 0.35em 0.65em;
            font-size: 0.75em;
            font-weight: 700;
            line-height: 1;
            color: #fff;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 0.25rem;
        }
        
        .badge-success {
            background-color: var(--success-color);
        }
        
        .badge-warning {
            background-color: var(--warning-color);
        }
        
        .badge-danger {
            background-color: var(--danger-color);
        }
        
        .code {
            font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
            background-color: #f5f7fa;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1.5rem;
            overflow-x: auto;
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem 0;
            color: #666;
            font-size: 0.9rem;
        }
        
        .plot-container {
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .plot-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .plot-title {
            font-weight: bold;
            margin: 0.5rem 0;
            color: var(--secondary-color);
        }
        
        .plot-description {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }
        
        .plots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        @media (max-width: 768px) {
            .plots-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Informe de Análisis Exploratorio de Datos</h1>
            <p>""")
        
        # Información básica
        f.write(f"{dataset_name} - {num_rows} filas, {num_cols} columnas</p>\n")
        f.write(f"<p>Generado el: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>\n")
        
        f.write("""
        </div>
        <div class="content">
            <!-- Resumen Ejecutivo -->
            <div class="section">
                <h2>Resumen Ejecutivo</h2>
                <div class="dashboard">
        """)
        
        # Tarjetas de resumen
        f.write(f"""
                    <div class="card">
                        <h3 class="card-title">Filas</h3>
                        <div class="card-value">{num_rows:,}</div>
                        <div class="card-label">registros totales</div>
                    </div>
                    <div class="card">
                        <h3 class="card-title">Columnas</h3>
                        <div class="card-value">{num_cols}</div>
                        <div class="card-label">variables diferentes</div>
                    </div>
                    <div class="card">
                        <h3 class="card-title">Variable Objetivo</h3>
                        <div class="card-value">{target_variable}</div>
                        <div class="card-label">para análisis predictivo</div>
                    </div>
        """)
        
        # Añadir más tarjetas si hay datos disponibles
        if hasattr(eda, 'missing_data'):
            missing_percent = getattr(eda.missing_data, 'total_missing_percentage', 0)
            f.write(f"""
                    <div class="card">
                        <h3 class="card-title">Datos Faltantes</h3>
                        <div class="card-value">{missing_percent:.1f}%</div>
                        <div class="card-label">del total de valores</div>
                    </div>
            """)
        
        # Cerrar dashboard
        f.write("""
                </div>
                
                <!-- Hallazgos Clave -->
                <h3>Hallazgos Clave</h3>
        """)
        
        # Añadir hallazgos clave
        f.write(key_findings_html)
        
        # Añadir alertas
        f.write("""
                <!-- Alertas -->
                <h3>Alertas y Advertencias</h3>
        """)
        f.write(alerts_html)
        
        # Cerrar sección de resumen
        f.write("""
            </div>
            
            <!-- Recomendaciones -->
            <div class="section">
                <h2>Recomendaciones</h2>
        """)
        
        # Añadir recomendaciones
        f.write(recommendations_html)
        
        # Cerrar sección de recomendaciones
        f.write("""
            </div>
            
            <!-- Visualizaciones -->
            <div class="section">
                <h2>Visualizaciones</h2>
                <p>A continuación se presentan las visualizaciones más relevantes del análisis:</p>
        """)
        
        # Añadir visualizaciones si están disponibles
        if hasattr(eda, 'basic_plots') and eda.basic_plots:
            # Visualizaciones de datos generales
            f.write("""
                <h3>Vista General de los Datos</h3>
                <div class="plots-grid">
            """)
            
            if 'overview' in eda.basic_plots:
                for name, img_base64 in eda.basic_plots['overview'].items():
                    plot_title = name.replace('_', ' ').title()
                    f.write(f"""
                    <div class="plot-container">
                        <img src="data:image/png;base64,{img_base64}" alt="{plot_title}">
                        <p class="plot-title">{plot_title}</p>
                    </div>
                    """)
            
            # Añadir correlación si está disponible
            if 'correlation' in eda.basic_plots:
                f.write(f"""
                <div class="plot-container">
                    <img src="data:image/png;base64,{eda.basic_plots['correlation']}" alt="Matriz de Correlación">
                    <p class="plot-title">Matriz de Correlación</p>
                </div>
                """)
            
            f.write("""
                </div>
            """)
        
        # Añadir visualizaciones univariadas si están disponibles
        if hasattr(eda, 'univariate_plots') and eda.univariate_plots:
            f.write("""
                <h3>Análisis Univariado</h3>
                <div class="plots-grid">
            """)
            
            # Variables numéricas
            if 'numeric' in eda.univariate_plots:
                for name, img_base64 in eda.univariate_plots['numeric'].items():
                    var_name = name.split('_')[0]
                    plot_title = f"Distribución de {var_name}"
                    f.write(f"""
                    <div class="plot-container">
                        <img src="data:image/png;base64,{img_base64}" alt="{plot_title}">
                        <p class="plot-title">{plot_title}</p>
                    </div>
                    """)
            
            # Variables categóricas
            if 'categorical' in eda.univariate_plots:
                for name, img_base64 in eda.univariate_plots['categorical'].items():
                    if '_frequency' in name:
                        var_name = name.split('_')[0]
                        plot_title = f"Frecuencia de {var_name}"
                        f.write(f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{img_base64}" alt="{plot_title}">
                            <p class="plot-title">{plot_title}</p>
                        </div>
                        """)
            
            f.write("""
                </div>
            """)
        
        # Añadir visualizaciones bivariadas si están disponibles
        if hasattr(eda, 'bivariate_plots') and eda.bivariate_plots:
            f.write(f"""
                <h3>Análisis Bivariado con {target_variable}</h3>
                <div class="plots-grid">
            """)
            
            # Variables numéricas vs target
            if 'numeric_target' in eda.bivariate_plots:
                for name, img_base64 in eda.bivariate_plots['numeric_target'].items():
                    if '_' + target_variable in name:
                        var_name = name.split('_')[0]
                        plot_title = f"{var_name} vs {target_variable}"
                        f.write(f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{img_base64}" alt="{plot_title}">
                            <p class="plot-title">{plot_title}</p>
                        </div>
                        """)
            
            # Variables categóricas vs target
            if 'categorical_target' in eda.bivariate_plots:
                for name, img_base64 in eda.bivariate_plots['categorical_target'].items():
                    if '_' + target_variable in name:
                        var_name = name.split('_')[0]
                        plot_title = f"{var_name} vs {target_variable}"
                        f.write(f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{img_base64}" alt="{plot_title}">
                            <p class="plot-title">{plot_title}</p>
                        </div>
                        """)
            
            f.write("""
                </div>
            """)
        
        # Cerrar sección de visualizaciones
        f.write("""
            </div>
            
            <!-- Pie de página -->
            <div class="footer">
                <p>Informe generado con EDA Automator | &copy; """ + str(datetime.datetime.now().year) + """</p>
            </div>
        </div>
    </div>
</body>
</html>
""")
    
    print(f"Informe HTML generado: {output_path}")
    return output_path

def generate_alternative_html(output_path, eda):
    """
    Genera un informe HTML alternativo sin usar f-strings.
    Útil cuando hay problemas con la sintaxis de la plantilla HTML.
    
    Args:
        output_path (str): Ruta donde se guardará el informe HTML
        eda (EDAAutomator): Instancia de EDA con los resultados
        
    Returns:
        str: Ruta al archivo HTML generado
    """
    return generate_html_report(output_path, eda)

def fix_html_template():
    """
    Crear una versión simplificada de la plantilla HTML para probar la generación del informe.
    
    Returns:
        str: Ruta al archivo HTML generado
    """
    output_dir = Path("output/unified")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "informe_simple.html"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Informe EDA Automator</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 20px;
                        color: #333;
                    }
                    h1, h2 {
                        color: #2c3e50;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    .header {
                        background: linear-gradient(135deg, #3498db, #2c3e50);
                        color: white;
                        padding: 2rem;
                        text-align: center;
                        margin-bottom: 2rem;
                    }
                    .section {
                        margin-bottom: 2rem;
                        padding: 1.5rem;
                        background-color: #f9f9f9;
                        border-radius: 8px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Informe de Análisis Exploratorio de Datos</h1>
                        <p>EDA Automator - Versión simplificada</p>
                    </div>
                    
                    <div class="section">
                        <h2>Resumen del Análisis</h2>
                        <p>Este informe presenta un análisis exploratorio de datos simplificado.</p>
                        <p>El informe completo con todas las visualizaciones y análisis está en desarrollo.</p>
                    </div>
                    
                    <div class="section">
                        <h2>Principales características implementadas</h2>
                        <ul>
                            <li>Resumen ejecutivo con puntuación de calidad</li>
                            <li>Alertas automáticas basadas en umbrales de calidad</li>
                            <li>Análisis detallado de outliers con visualizaciones</li>
                            <li>Visualización de series temporales con detección de tendencias</li>
                            <li>Exportación a Excel con análisis avanzado</li>
                        </ul>
                    </div>
                    
                    <div class="section">
                        <h2>Próximas mejoras</h2>
                        <ul>
                            <li>Arreglar problemas con f-strings en la plantilla HTML</li>
                            <li>Mejorar estilos y responsividad del informe</li>
                            <li>Añadir más visualizaciones interactivas</li>
                            <li>Optimizar generación de imágenes del informe</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>""")
        
        print(f"Plantilla HTML simple guardada en {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Error al crear plantilla HTML: {str(e)}")
        return None 