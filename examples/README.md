# EDA Automator - Ejemplos

Este directorio contiene ejemplos de uso de EDA Automator para demostrar diferentes funcionalidades.

## Contenido

- `unified_eda_modular.py`: Script que muestra cómo utilizar el módulo unificado para generar reportes en varios formatos
- `notebooks/`: Jupyter notebooks con ejemplos detallados de uso

## Ejemplos de uso

### Generación de reportes con el módulo unificado

```bash
# Generar reportes en todos los formatos disponibles
python examples/unified_eda_modular.py --formats all

# Generar solo un reporte HTML
python examples/unified_eda_modular.py --formats html --size 2000

# Generar un reporte con dataset de series temporales
python examples/unified_eda_modular.py --data-type timeseries --output output/timeseries
```

### Uso desde línea de comandos

```bash
# Analizar un conjunto de datos existente
eda-automator analyze data.csv --formats html markdown

# Generar un conjunto de datos sintético
eda-automator generate --size 5000 --output test_data.csv
```

## Estructura del directorio

```
examples/
├── unified_eda_modular.py     # Script para usar el módulo unificado
├── notebooks/                 # Jupyter notebooks
│   ├── quickstart.ipynb       # Introducción rápida
│   └── advanced_usage.ipynb   # Ejemplos avanzados
└── README.md                  # Este archivo
```

Para más información, consulte la documentación completa en el README principal. 