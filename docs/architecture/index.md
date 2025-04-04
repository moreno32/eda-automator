# EDA Automator Architecture

Este documento proporciona una visión general de la arquitectura de EDA Automator, explicando las decisiones de diseño, las interacciones entre componentes y los detalles de implementación.

## Visión General de la Arquitectura

EDA Automator sigue una arquitectura modular diseñada en torno a una clara separación de responsabilidades, haciendo que el código sea mantenible, extensible y testeable. La arquitectura se organiza alrededor de los siguientes principios fundamentales:

- **Principio de Responsabilidad Única**: Cada módulo tiene una responsabilidad claramente definida
- **Separación de Preocupaciones**: Análisis, visualización e informes son preocupaciones distintas
- **Configurabilidad**: Los componentes pueden configurarse a través de un sistema de configuración centralizado
- **Extensibilidad**: Se pueden añadir fácilmente nuevos tipos de análisis, visualizaciones y formatos de informes

## Estructura de Módulos

El paquete EDA Automator está organizado en la siguiente estructura:

```
eda_automator/
├── core/                     # Módulo principal con funcionalidad central
│   ├── analysis/             # Módulos de análisis
│   │   ├── __init__.py       # Inicialización del módulo
│   │   ├── basic.py          # Análisis básico del dataset
│   │   ├── missing.py        # Análisis de valores faltantes
│   │   ├── outliers.py       # Detección de valores atípicos
│   │   ├── correlation.py    # Análisis de correlación
│   │   ├── distribution.py   # Análisis de distribución
│   │   └── target.py         # Análisis de variable objetivo
│   ├── data/                 # Manejo de datos
│   │   ├── __init__.py       # Inicialización del módulo
│   │   ├── loader.py         # Funciones para cargar datos
│   │   └── generator.py      # Generación de datasets
│   ├── report_generators/    # Generación de informes
│   │   ├── __init__.py       # Inicialización del módulo
│   │   ├── html.py           # Generación de informes HTML
│   │   ├── markdown.py       # Generación de documentos Markdown
│   │   └── image.py          # Generación de imágenes de informes
│   ├── utils/                # Funciones de utilidad
│   │   ├── __init__.py       # Inicialización del módulo
│   │   ├── formatting.py     # Formato de strings y valores
│   │   ├── logger.py         # Utilidades de logging
│   │   ├── environment.py    # Configuración del entorno
│   │   └── dependencies.py   # Verificación de dependencias
│   ├── visualization/        # Funciones de visualización
│   │   ├── __init__.py       # Inicialización del módulo
│   │   ├── basic.py          # Visualizaciones básicas
│   │   ├── distribution.py   # Gráficos de distribución
│   │   ├── correlation.py    # Visualizaciones de correlación
│   │   └── target.py         # Visualizaciones relacionadas con la variable objetivo
│   ├── templates/            # Plantillas de informes
│   │   ├── default.html      # Plantilla HTML predeterminada
│   │   └── default.css       # Estilos CSS predeterminados
│   ├── __init__.py           # Inicialización del módulo principal
│   ├── automator.py          # Clase principal EDACore
│   └── config.py             # Gestión de configuración
├── unified/                  # Módulo unificado (compatibilidad con versiones anteriores)
├── requirements/             # Archivos de requisitos para dependencias
│   └── image_generation.txt  # Requisitos para generación de imágenes
├── __init__.py               # Inicialización del paquete
└── cli.py                    # Interfaz de línea de comandos
```

## Componentes Principales

### Clase EDACore

La clase `EDACore` en `core/automator.py` es el componente central de la arquitectura. Sus responsabilidades incluyen:

- Inicializar el flujo de análisis con un DataFrame y configuración
- Gestionar la ejecución de diferentes módulos de análisis
- Coordinar la validación y transformación de datos
- Manejar la generación de visualizaciones
- Facilitar la creación de informes

```python
from eda_automator.core import EDACore

# Inicializar con un dataframe
eda = EDACore(
    dataframe=df, 
    target_variable='target',
    settings={
        'sampling_threshold': 5000,
        'correlation_method': 'spearman'
    }
)

# Ejecutar análisis completo
results = eda.run_full_analysis()

# Generar visualizaciones
figures = eda.generate_visualizations()

# Generar informe
eda.generate_report(output_path='eda_report.html', format='html')
```

### Módulos de Análisis

El directorio `core/analysis/` contiene módulos especializados para diferentes tipos de análisis:

- `basic.py`: Proporciona información general del dataset y estadísticas
- `missing.py`: Analiza patrones y distribuciones de valores faltantes
- `outliers.py`: Detecta valores atípicos utilizando varios métodos estadísticos
- `correlation.py`: Analiza relaciones entre variables
- `distribution.py`: Examina las características de distribución de las variables
- `target.py`: Analiza la relación entre características y una variable objetivo

Cada módulo de análisis sigue un patrón de interfaz consistente para la integración con la clase principal `EDACore`.

### Manejo de Datos

El módulo `core/data/` proporciona funcionalidad para cargar y generar datos:

- `loader.py`: Funciones para cargar datos de varias fuentes, incluyendo CSV, Excel, y bases de datos
- `generator.py`: Funciones para generar datasets sintéticos para pruebas y demostración

### Sistema de Visualización

El módulo `core/visualization/` contiene funciones para crear visualizaciones estandarizadas:

- `basic.py`: Visualizaciones de información general
- `distribution.py`: Gráficos de análisis de distribución
- `correlation.py`: Visualizaciones de correlación
- `target.py`: Visualizaciones relacionadas con la variable objetivo

Todas las funciones de visualización siguen un estilo consistente y admiten muestreo para grandes conjuntos de datos.

### Generadores de Informes

El módulo `core/report_generators/` maneja la generación de informes en diferentes formatos:

- `html.py`: Crea informes HTML interactivos
- `markdown.py`: Genera documentos Markdown
- `image.py`: Produce exportaciones de informes en formato imagen

Cada generador de informes utiliza plantillas y formato estandarizado para garantizar la consistencia.

### Sistema de Configuración

El sistema de configuración en `core/config.py` proporciona:

- Valores de configuración predeterminados
- Validación de configuración
- Fusión de configuraciones de múltiples fuentes
- Propagación de configuraciones a todos los componentes

## Flujo de Datos e Interacción de Componentes

El flujo de datos en EDA Automator sigue estos pasos, con interacciones detalladas entre componentes:

1. **Inicialización**:
   - El usuario crea una instancia de `EDACore` con un DataFrame
   - `EDACore` valida el DataFrame y configura el entorno
   - Se inicializa el sistema de registro (logging) y se verifican dependencias

2. **Carga y Preparación de Datos**:
   - Los datos pueden ser cargados mediante `data/loader.py` o proporcionados directamente
   - Se realizan comprobaciones iniciales de tipos de datos y estructura
   - Se aplica muestreo automático si es necesario según configuración

3. **Ejecución de Análisis**:
   - `EDACore.run_full_analysis()` orquesta múltiples módulos de análisis
   - Cada módulo de análisis recibe el DataFrame y configuración
   - Los resultados se recopilan en una estructura de datos unificada
   - Diagrama de secuencia de análisis:
     ```
     EDACore → basic_analysis → missing_values_analysis → outlier_analysis 
            → correlation_analysis → distribution_analysis → target_analysis
     ```

4. **Generación de Visualizaciones**:
   - `EDACore.generate_visualizations()` coordina la creación de figuras
   - Cada módulo de visualización corresponde a un tipo de análisis
   - Las visualizaciones utilizan estilos y paletas consistentes
   - Se aplica muestreo inteligente para conjuntos de datos grandes

5. **Creación de Informes**:
   - `EDACore.generate_report()` envía datos, resultados y figuras al generador apropiado
   - El generador de informes seleccionado construye la salida en el formato deseado
   - Se utilizan plantillas para mantener la consistencia visual
   - Los informes incluyen metadatos, configuraciones y código reproducible

El siguiente diagrama muestra un flujo de datos simplificado:

```
┌─────────────┐     ┌─────────────┐     ┌───────────────┐
│  DataFrame  │────▶│  EDACore    │────▶│ Módulos de    │
└─────────────┘     │ Inicializa- │     │  Análisis     │
                    │   ción      │     └───────┬───────┘
┌─────────────┐     └──────┬──────┘             │
│ Configura-  │────────────┘                    ▼
│   ción      │                        ┌───────────────┐
└─────────────┘                        │  Resultados   │
                                       │ de Análisis   │
                                       └───────┬───────┘
┌─────────────┐     ┌─────────────┐            │
│ Plantillas  │────▶│ Generadores │◀───────────┘
└─────────────┘     │ de Informes │            ▲
                    └──────┬──────┘            │
                           │           ┌───────────────┐
                           ▼           │ Funciones de  │
                    ┌─────────────┐    │ Visualización │
                    │  Informes   │◀───┴───────────────┘
                    └─────────────┘
```

## Puntos de Extensión

EDA Automator está diseñado para ser extendido de varias maneras:

1. **Nuevos Tipos de Análisis**: 
   - Añadir módulos al directorio `core/analysis/`
   - Implementar la función de análisis con la interfaz estándar 
   - Actualizar `EDACore` para invocar el nuevo análisis

2. **Visualizaciones Personalizadas**: 
   - Extender los módulos `core/visualization/`
   - Garantizar la coherencia con el estilo existente
   - Vincular con los módulos de análisis correspondientes

3. **Formatos de Informes Adicionales**: 
   - Implementar nuevos generadores en `core/report_generators/`
   - Seguir el patrón de interfaz establecido
   - Actualizar `EDACore.generate_report()` para soportar el nuevo formato

4. **Plantillas Personalizadas**: 
   - Crear plantillas personalizadas en el directorio `core/templates/`
   - Permitir la selección de plantillas a través de parámetros de configuración

## Interfaz de Línea de Comandos

La interfaz de línea de comandos en `cli.py` proporciona una forma conveniente de usar EDA Automator sin escribir código Python:

```bash
# Analizar un dataset
eda-automator analyze data.csv -o reports -f html -t target

# Generar un dataset sintético
eda-automator dataset -o generated_data.csv -s 1000 -c 10 -t basic

# Usar configuraciones personalizadas
eda-automator analyze data.csv -o reports -f html -c custom_config.yaml

# Generar múltiples formatos de informes
eda-automator analyze data.csv -o reports -f html,markdown,image
```

## Compatibilidad con Versiones Anteriores

Para mantener compatibilidad con versiones anteriores, el directorio `unified/` contiene la implementación monolítica heredada. Las nuevas características se implementan en la arquitectura modular, pero el código heredado se mantiene para los usuarios existentes.

El sistema de envoltura (wrapper) permite que los usuarios de la API antigua interactúen con la nueva arquitectura de forma transparente:

```python
# Uso de API antigua
from eda_automator import EDAAutomator  # Envoltura que utiliza EDACore internamente

# Funciona igual que antes
automator = EDAAutomator(dataframe=df)
automator.run_full_analysis()
automator.generate_report("report.html")
```

## Consideraciones de Rendimiento

EDA Automator incluye varias optimizaciones de rendimiento:

1. **Muestreo Automático**: 
   - Los grandes conjuntos de datos se muestrean automáticamente según la configuración
   - Se utilizan técnicas de muestreo estratificado para preservar distribuciones

2. **Almacenamiento en Caché**: 
   - Los resultados intermedios se almacenan en caché para evitar cálculos redundantes
   - Implementación eficiente para resultados costosos como matrices de correlación

3. **Evaluación Perezosa (Lazy Evaluation)**: 
   - Las visualizaciones se generan solo cuando son necesarias
   - Ahorra recursos computacionales para análisis parciales

4. **Procesamiento Paralelo**: 
   - Múltiples análisis pueden ejecutarse en paralelo cuando es apropiado
   - Aprovecha eficientemente los recursos de hardware disponibles

5. **Optimización de Memoria**:
   - Utiliza operaciones vectorizadas de pandas para manipulación eficiente
   - Implementa liberación de memoria para grandes conjuntos de datos

## Manejo de Errores

La arquitectura incluye un manejo integral de errores:

1. **Validación de Entrada**: 
   - Todas las entradas se validan tempranamente para prevenir problemas
   - Comprobaciones exhaustivas de tipo y formato de datos

2. **Degradación Elegante**: 
   - Si un componente falla, otros continúan funcionando
   - El sistema de informes puede adaptarse a análisis parciales

3. **Registro Detallado**: 
   - Los errores se registran con contexto para depuración
   - Sistema de nivel de registro configurable (debug, info, warning, error)

4. **Retroalimentación al Usuario**: 
   - Mensajes de error claros ayudan a los usuarios a resolver problemas
   - Sugerencias útiles para errores comunes

## Direcciones Futuras de la Arquitectura

Las mejoras arquitectónicas planificadas incluyen:

1. **Sistema de Plugins**: 
   - Soporte para plugins de terceros y extensiones
   - API estable para desarrolladores externos

2. **Procesamiento Distribuido**: 
   - Soporte para procesar grandes conjuntos de datos a través de múltiples nodos
   - Integración con frameworks como Dask o Spark

3. **Interfaz Web Interactiva**: 
   - UI basada en web para análisis interactivo
   - Visualizaciones dinámicas y filtrado en tiempo real

4. **Colaboración en Tiempo Real**: 
   - Soporte para análisis colaborativo y anotación
   - Compartir y comentar informes interactivamente

5. **Integración de IA**: 
   - Generación automática de insights basados en patrones detectados
   - Recomendaciones inteligentes para análisis adicionales
