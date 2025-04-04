# EDA Automator

Un paquete para automatizar el análisis exploratorio de datos (EDA) y generar reportes interactivos en varios formatos.

## Características

- **Análisis automático** de conjuntos de datos tabulares
- **Visualizaciones interactivas** para explorar patrones en los datos
- **Reportes en múltiples formatos**: HTML, Markdown, Excel, e imágenes
- **Interfaz de línea de comandos** para análisis rápidos
- **API Python** para integración en flujos de trabajo existentes
- **Modular y extensible** para personalizar el análisis

## Instalación

```bash
# Instalación básica
pip install eda-automator

# Con todas las dependencias para generación de imágenes
pip install eda-automator[image]
```

## Uso rápido

### Desde línea de comandos

```bash
# Analizar un dataset y generar reporte HTML
eda-automator analyze data.csv --output reports

# Generar reportes en múltiples formatos
eda-automator analyze data.csv --formats html markdown landscape excel

# Generar un dataset sintético para pruebas
eda-automator generate --size 1000 --output test_data.csv
```

### Desde Python

```python
import pandas as pd
from eda_automator import EDAAutomator, run_analysis

# Cargar datos
df = pd.read_csv('data.csv')

# Método simplificado
eda = run_analysis(df, target_variable='target')

# Generar reporte
eda.generate_html_report('report.html')

# O usar la clase EDAAutomator para más control
eda = EDAAutomator(df)
eda.run_basic_analysis()
eda.run_missing_analysis()
eda.run_outlier_analysis()
eda.run_correlation_analysis()
eda.generate_html_report('custom_report.html')
```

## Módulo Unificado

El paquete incluye un módulo unificado que simplifica el proceso de análisis:

```python
from eda_automator.unified import (
    load_data,
    create_dataset,
    run_analysis,
    setup_environment
)
from eda_automator.unified.report_generators import (
    generate_html_report,
    generate_markdown_report
)

# Configurar el entorno (opcional)
setup_environment(language='es')

# Cargar o generar datos
df = load_data('data.csv')  # o create_dataset(size=1000)

# Analizar los datos
eda = run_analysis(df, target_variable='target')

# Generar reportes
generate_html_report('report.html', eda)
generate_markdown_report('report.md', eda)
```

## Dependencias Opcionales

Para generar reportes en formato imagen, instale una de las siguientes combinaciones:

- `imgkit` + `wkhtmltopdf` (recomendado): `pip install imgkit`
- `selenium` + `webdriver para Chrome`: `pip install selenium webdriver-manager`
- `weasyprint` + `pdf2image`: `pip install weasyprint pdf2image`

## Contribución

Las contribuciones son bienvenidas. Por favor, abra un issue o pull request.

## Licencia

MIT License

## 🌟 Características

- **Análisis Automático**: Detecta y analiza automáticamente características importantes de tus datos.
- **Visualizaciones de Alta Calidad**: Genera gráficos y visualizaciones informativas para datos categóricos y numéricos.
- **Informes Multi-formato**: Crea informes en HTML, Markdown o imagen.
- **Estructura Modular**: Arquitectura modular para facilitar la extensión y personalización.
- **Detección Inteligente**: Identificación automática de outliers, relaciones y patrones.

## 🚀 Instalación

```bash
pip install eda-automator
```

## 📊 Uso Rápido

### Enfoque Modular (Recomendado)

```python
from eda_automator import EDAAutomator
import pandas as pd

# Cargar tus datos
df = pd.read_csv('mi_dataset.csv')

# Crear una instancia de EDA Automator
eda = EDAAutomator(df, target_variable='mi_variable_objetivo')

# Ejecutar análisis completo
eda.run_full_analysis()

# Generar informe
eda.generate_report('informe_eda.html', format='html')
```

### Enfoque mediante Línea de Comandos

```bash
# Ejemplo básico con dataset sintético
python examples/unified_eda_modular.py --size 1000 --formats html

# Especificar tipo de datos y directorio de salida
python examples/unified_eda_modular.py --data-type timeseries --output output/mi_informe

# Ver opciones disponibles
python examples/unified_eda_modular.py --help
```

## 📝 Ejemplos

### Jupyter Notebook de Inicio Rápido
Consulta el [notebook de inicio rápido](examples/notebooks/quickstart.ipynb) para obtener ejemplos prácticos.

### Ejemplos de Línea de Comandos
Prueba los scripts de ejemplo en el directorio `examples/`:

```bash
python examples/unified_eda_modular.py --formats html markdown --size 1000
```

## 🧩 Estructura del Proyecto

```
eda_automator/                 # Paquete principal
├── unified/                   # Módulo unificado centralizado
│   ├── report_generators/     # Generadores de reportes unificados
│   ├── __init__.py            # Inicialización del módulo unificado
│   ├── analysis.py            # Análisis de datos
│   ├── config.py              # Configuración
│   ├── data.py                # Manejo de datos
│   ├── dependencies.py        # Verificación de dependencias
│   ├── main.py                # Funcionalidad principal
│   ├── utils.py               # Utilidades
│   └── visualizations.py      # Visualizaciones
├── report_generators/         # Generadores de reportes clásicos
├── __init__.py                # Inicialización del paquete
├── bivariate.py               # Análisis bivariado
├── cli.py                     # Interfaz de línea de comandos
├── data_quality.py            # Análisis de calidad de datos
├── multivariate.py            # Análisis multivariado
├── report.py                  # Generación de reportes
├── stats_analysis.py          # Análisis estadístico
├── univariate.py              # Análisis univariado
├── utils.py                   # Funciones de utilidad
└── visuals.py                 # Visualizaciones

examples/                      # Ejemplos de uso
├── notebooks/                 # Notebooks de ejemplo
├── unified_eda_modular.py     # Script de ejemplo para módulo unificado
└── README.md                  # Documentación de ejemplos
```

## 📄 Licencia

Este proyecto está licenciado bajo la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- A todos los colaboradores y usuarios de EDA Automator.
- A la comunidad de ciencia de datos por su continua inspiración y apoyo. 