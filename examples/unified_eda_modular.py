"""
EDA Automator - Ejemplo de uso del módulo unificado

Este script muestra cómo utilizar el módulo unificado del paquete EDA Automator
para generar reportes de análisis exploratorio de datos en varios formatos.
"""

import os
import sys
import argparse
from pathlib import Path

# Añadir el directorio padre al path para acceder al paquete principal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar desde el módulo unificado
from eda_automator.unified.report_generators import (
    generate_html_report, 
    generate_markdown_report,
    generate_landscape_report,
    generate_portrait_report,
    AVAILABLE_FORMATS
)
from eda_automator.unified.data import create_dataset
from eda_automator.unified.dependencies import check_dependencies
from eda_automator.unified.analysis import run_analysis
from eda_automator.unified.config import setup_environment

def generate_reports(eda, output_dir_path, formats=None):
    """
    Generar reportes en los formatos especificados.

    Args:
        eda (EDAAutomator): Instancia con resultados del análisis
        output_dir_path (str): Directorio para guardar los reportes
        formats (list): Lista de formatos de salida (html, markdown, landscape, portrait, excel)

    Returns:
        list: Lista de tuplas (formato, ruta) de los reportes generados
    """
    if formats is None:
        formats = ['html']
    
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reports_generated = []
    
    # Generar reporte HTML
    if 'html' in formats:
        print("\nGenerando reporte HTML...")
        html_path = str(output_dir / "full_report.html")
        html_file = generate_html_report(html_path, eda)
        reports_generated.append(('HTML', html_file))
        print(f"Reporte HTML guardado en {html_file}")
    else:
        # Si necesitamos HTML para generación de imágenes posteriormente
        html_path = str(output_dir / "full_report.html")
    
    # Generar reporte Markdown
    if 'markdown' in formats:
        print("\nGenerando reporte Markdown...")
        md_path = str(output_dir / "full_report.md")
        md_file = generate_markdown_report(md_path, eda)
        reports_generated.append(('Markdown', md_file))
        print(f"Reporte Markdown guardado en {md_file}")
    
    # Generar reporte Excel (si está disponible)
    if 'excel' in formats and 'excel' in AVAILABLE_FORMATS:
        print("\nGenerando reporte Excel...")
        excel_path = str(output_dir / "full_report.xlsx")
        try:
            excel_file = AVAILABLE_FORMATS['excel']['function'](excel_path, eda)
            reports_generated.append(('Excel', excel_file))
            print(f"Reporte Excel guardado en {excel_file}")
        except Exception as e:
            print(f"Error al generar reporte Excel: {str(e)}")
            print("La generación de reportes Excel puede requerir dependencias adicionales.")
    
    # Asegurarse de que haya un reporte HTML para convertir
    if not os.path.exists(html_path) and ('landscape' in formats or 'portrait' in formats):
        print("\nGenerando reporte HTML para conversión a imagen...")
        html_file = generate_html_report(html_path, eda)
        html_path = html_file
    
    # Generar imagen horizontal
    if 'landscape' in formats and os.path.exists(html_path):
        print("\nGenerando reporte de imagen en formato horizontal...")
        output_path = str(output_dir / "full_report_landscape.png")
        landscape_path = generate_landscape_report(output_path, html_path, eda)
        if landscape_path:
            reports_generated.append(('Imagen horizontal', landscape_path))
            print(f"Reporte de imagen horizontal guardado en {landscape_path}")
    
    # Generar imagen vertical
    if 'portrait' in formats and os.path.exists(html_path):
        print("\nGenerando reporte de imagen en formato vertical...")
        output_path = str(output_dir / "full_report_portrait.png")
        portrait_path = generate_portrait_report(output_path, html_path, eda)
        if portrait_path:
            reports_generated.append(('Imagen vertical', portrait_path))
            print(f"Reporte de imagen vertical guardado en {portrait_path}")
    
    return reports_generated

def main():
    """Función principal para ejecutar el script."""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Generador de reportes EDA unificado')
    parser.add_argument('--size', type=int, default=1000, help='Tamaño del dataset')
    parser.add_argument('--language', choices=['es', 'en'], default='es', help='Idioma del reporte')
    parser.add_argument('--output', default='output/unified', help='Directorio de salida')
    parser.add_argument('--data-type', choices=['basic', 'timeseries'], default='basic', 
                       help='Tipo de dataset')
    parser.add_argument('--formats', nargs='+', 
                       choices=['html', 'markdown', 'landscape', 'portrait', 'excel', 'all'],
                       default=['all'], help='Formatos de salida')
    
    args = parser.parse_args()
    
    # Si se elige 'all', incluir todos los formatos
    if 'all' in args.formats:
        formats = ['html', 'markdown', 'landscape', 'portrait', 'excel']
    else:
        formats = args.formats
    
    # Verificar dependencias si se van a generar imágenes
    if 'landscape' in formats or 'portrait' in formats:
        dependencies = check_dependencies()
        if not any([dependencies['imgkit'], dependencies['selenium'], dependencies['weasyprint']]):
            print("\nADVERTENCIA: No se encontraron herramientas para generación de imágenes.")
            print("Instale una de las siguientes dependencias:")
            print("  - imgkit + wkhtmltopdf")
            print("  - selenium + webdriver para Chrome")
            print("  - weasyprint + pdf2image")
            print("\nContinuando con los formatos disponibles...\n")
            # Eliminar formatos que requieren herramientas externas
            formats = [f for f in formats if f not in ['landscape', 'portrait']]
    
    # Configurar el entorno
    setup_environment(language=args.language)
    
    # Crear dataset
    df = create_dataset(size=args.size, data_type=args.data_type)
    
    # Elegir variable objetivo apropiada según el tipo de datos
    target_variable = 'income_ts' if args.data_type == 'timeseries' else 'churn'
    
    # Ejecutar análisis
    eda = run_analysis(df, target_variable=target_variable)
    
    # Generar reportes
    reports = generate_reports(eda, args.output, formats=formats)
    
    # Mostrar resumen final
    if reports:
        print("\nReportes generados:")
        for report_format, report_path in reports:
            print(f"  - {report_format}: {report_path}")
    else:
        print("\nNo se generaron reportes.")

if __name__ == "__main__":
    print("=== EDA Automator - Generador de reportes unificado ===\n")
    main() 