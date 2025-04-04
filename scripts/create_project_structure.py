import os
import pathlib

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path, content=""):
    """Create a file with the given content."""
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def create_project_structure():
    """Create the EDA Automator project structure."""
    # Project root
    project_root = pathlib.Path(".")
    
    # Main package directory
    package_dir = project_root / "eda_automator"
    create_directory(package_dir)
    
    # Test directory
    test_dir = package_dir / "tests"
    create_directory(test_dir)
    
    # Create main module files
    module_files = [
        "__init__.py",
        "data_quality.py",
        "stats_analysis.py",
        "univariate.py",
        "bivariate.py",
        "multivariate.py",
        "visuals.py",
        "utils.py",
        "report.py"
    ]
    
    # Basic placeholder content for Python files
    basic_content = '''"""
{} module for EDA Automator.

This module is part of the EDA Automator package.
"""

# Placeholder for {} module implementation
'''
    
    for module in module_files:
        module_name = module.replace('.py', '')
        content = basic_content.format(
            module_name.replace('_', ' ').title(), 
            module_name
        ) if module.endswith('.py') else ""
        create_file(package_dir / module, content)
    
    # Create configuration file
    config_content = '''# EDA Automator Configuration
# Default thresholds and parameters

# Data Quality Parameters
missing_threshold: 0.2
outlier_threshold: 3.0
outlier_method: "z-score"  # Options: "z-score", "iqr"

# Visualization Parameters
categorical_threshold: 10
max_unique_values: 20
sampling_threshold: 10000

# Correlation Analysis
correlation_threshold: 0.7
correlation_method: "pearson"  # Options: "pearson", "spearman", "kendall"
'''
    create_file(package_dir / "config.yaml", config_content)
    
    # Create test files
    test_files = [
        "__init__.py",
        "test_data_quality.py",
        "test_stats_analysis.py",
        "test_univariate.py",
        "test_bivariate.py",
        "test_multivariate.py",
        "test_utils.py"
    ]
    
    test_content = '''"""
Tests for the {} module.
"""
import unittest
import pandas as pd
from eda_automator import {}


class Test{}(unittest.TestCase):
    """Test case for {} module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "categorical": ["A", "B", "A", "C", "B"],
            "missing": [1, None, 3, None, 5]
        })

    def test_sample(self):
        """Sample test method."""
        self.assertTrue(True)  # Placeholder assertion


if __name__ == "__main__":
    unittest.main()
'''
    
    for test_file in test_files:
        if test_file == "__init__.py":
            create_file(test_dir / test_file, "")
        else:
            module_name = test_file.replace('test_', '').replace('.py', '')
            class_name = module_name.replace('_', ' ').title().replace(' ', '')
            content = test_content.format(
                module_name, 
                module_name,
                class_name,
                module_name
            )
            create_file(test_dir / test_file, content)
    
    # Create project files
    pyproject_content = '''[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ["py38"]
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
'''
    
    setup_content = '''from setuptools import setup, find_packages

setup(
    name="eda_automator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
        "seaborn>=0.10.0",
        "scikit-learn>=0.22.0",
        "scipy>=1.4.0",
        "statsmodels>=0.11.0",
        "pyyaml>=5.3.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for automated exploratory data analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/eda_automator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
)
'''
    
    readme_content = '''# EDA Automator

A Python library for automated exploratory data analysis.

## Overview

EDA Automator helps data scientists and analysts perform quick and comprehensive exploratory data analysis with minimal code. The library automates common EDA tasks such as:

- Data quality assessment (missing values, duplicates, outliers)
- Statistical analysis (descriptive statistics, normality tests)
- Visualization (univariate, bivariate, and multivariate analysis)
- Reporting (comprehensive HTML/Markdown reports)

## Installation

```bash
pip install eda-automator
```

## Quick Start

```python
import pandas as pd
from eda_automator import EDAAutomator

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize and run analysis
automator = EDAAutomator(data, target_variable="target_column")
report = automator.run_full_analysis()

# Generate a report
report.to_html("eda_report.html")
```

## Features

- **Data Quality Analysis**: Automatically detect missing values, duplicates, and outliers
- **Statistical Analysis**: Calculate descriptive statistics and run normality tests
- **Visualization**: Generate professional visualizations for univariate, bivariate, and multivariate analysis
- **Reporting**: Create comprehensive reports that consolidate all findings

## Documentation

For full documentation, visit [documentation link].

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
'''
    
    create_file(project_root / "pyproject.toml", pyproject_content)
    create_file(project_root / "setup.py", setup_content)
    create_file(project_root / "README.md", readme_content)
    
    # Create GitHub Actions workflow file
    github_dir = project_root / ".github" / "workflows"
    create_directory(github_dir)
    
    workflow_content = '''name: Python Package

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 black isort
        if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - name: Check formatting with black
      run: |
        black --check .
    - name: Check imports with isort
      run: |
        isort --check-only --profile black .

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9']
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
        if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
    - name: Test with pytest
      run: |
        pytest --cov=eda_automator tests/

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: |
        python -m build
    - name: Check package
      run: |
        twine check dist/*
'''
    
    create_file(github_dir / "python-package.yml", workflow_content)
    
    # Create a .mdc file for Cursor best practices
    cursor_dir = project_root / ".cursor" / "rules"
    create_directory(cursor_dir)
    
    mdc_content = '''---
description: "Reglas y lineamientos principales para el proyecto EDA Automator"
globs: ["*.py", "*.ipynb"]
alwaysApply: true
---

# 1. Estilo de Código y Modularidad

- "Cada módulo debe dedicarse a una sola responsabilidad, según la estructura definida."
- "Usa nombres descriptivos y sigue las convenciones PEP8 para el código."

# 2. Documentación y Docstrings

- "Utiliza estilo NumPy para los docstrings en todas las funciones y clases."
- "Incluye ejemplos en la documentación para facilitar la comprensión."

# 3. Logs y Excepciones

- "Usa logging en lugar de print() para mensajes informativos."
- "Maneja excepciones de forma apropiada con mensajes descriptivos."

# 4. Testing y Cobertura

- "Cada módulo debe tener un archivo de test con casos básicos y extremos."
- "Mantén una cobertura de pruebas superior al 80%."

# 5. Visualizaciones y Reportes

- "Centraliza los estilos de gráficos en visuals.py para mantener coherencia."
- "Usa paletas de colores accesibles y profesionales."

# 6. Buenas Prácticas Generales

- "Revisa y optimiza el código para datasets grandes aplicando muestreo cuando sea necesario."
- "Mantén la modularidad permitiendo reemplazar componentes (por ejemplo, métodos de detección de outliers)."
'''
    create_file(cursor_dir / "eda_automator_guidelines.mdc", mdc_content)
    
    print("\nProject structure created successfully!")
    print("""
Phase 1 of the roadmap completed, which includes:
- Structure and Repository setup
- Configuration of GitHub Actions workflows
- Documentation initial setup
- Project configuration file
- Setup Cursor rules guidelines
""")

if __name__ == "__main__":
    create_project_structure() 