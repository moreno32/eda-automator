"""
Generadores de reportes para EDA Automator (versión unificada).

Este módulo contiene funciones para generar reportes de análisis exploratorio de datos
en varios formatos, incluyendo HTML, Excel, Markdown e imágenes.
"""

from .html_generator import generate_html_report, generate_alternative_html
from .markdown_generator import generate_markdown_report
from .image_generator import generate_landscape_report, generate_portrait_report

# Importaciones condicionales para Excel (requieren dependencias adicionales)
try:
    from .excel_generator import generate_excel_report
    __all__ = [
        'generate_html_report',
        'generate_alternative_html',
        'generate_excel_report',
        'generate_markdown_report',
        'generate_landscape_report',
        'generate_portrait_report'
    ]
except ImportError:
    __all__ = [
        'generate_html_report',
        'generate_alternative_html',
        'generate_markdown_report',
        'generate_landscape_report',
        'generate_portrait_report'
    ]

# Formatos de reportes disponibles
AVAILABLE_FORMATS = {
    'html': {
        'description': 'Reporte HTML interactivo',
        'function': generate_html_report
    },
    'markdown': {
        'description': 'Reporte Markdown con gráficos incrustados',
        'function': generate_markdown_report
    },
    'landscape': {
        'description': 'Reporte en imagen con orientación horizontal',
        'function': generate_landscape_report
    },
    'portrait': {
        'description': 'Reporte en imagen con orientación vertical',
        'function': generate_portrait_report
    }
}

# Añadir Excel si está disponible
try:
    from .excel_generator import generate_excel_report
    AVAILABLE_FORMATS['excel'] = {
        'description': 'Libro de Excel con múltiples pestañas',
        'function': generate_excel_report
    }
except ImportError:
    pass 