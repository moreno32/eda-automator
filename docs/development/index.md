# Developer Documentation

This section provides technical documentation for developers who want to contribute to the EDA Automator project or extend its functionality.

## Getting Started with Development

### Setting Up Your Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/eda_automator.git
   cd eda_automator
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Project Structure

The EDA Automator project follows a modular architecture:

```
eda_automator/
├── core/                     # Core module with main functionality
│   ├── analysis/             # Analysis modules
│   │   ├── __init__.py       # Module initialization
│   │   ├── basic.py          # Basic dataset analysis
│   │   ├── missing.py        # Missing values analysis
│   │   ├── outliers.py       # Outlier detection
│   │   ├── correlation.py    # Correlation analysis
│   │   ├── distribution.py   # Distribution analysis
│   │   └── target.py         # Target variable analysis
│   ├── data/                 # Data handling
│   │   ├── __init__.py       # Module initialization
│   │   ├── loader.py         # Data loading functions
│   │   └── generator.py      # Dataset generation
│   ├── report_generators/    # Report generation
│   │   ├── __init__.py       # Module initialization
│   │   ├── html.py           # HTML report generation
│   │   ├── markdown.py       # Markdown report generation
│   │   └── image.py          # Image report generation
│   ├── utils/                # Utility functions
│   │   ├── __init__.py       # Module initialization
│   │   ├── formatting.py     # String and value formatting
│   │   ├── logger.py         # Logging utilities
│   │   ├── environment.py    # Environment setup
│   │   └── dependencies.py   # Dependency checking
│   ├── visualization/        # Visualization functions
│   │   ├── __init__.py       # Module initialization
│   │   ├── basic.py          # Basic visualizations
│   │   ├── distribution.py   # Distribution plots
│   │   ├── correlation.py    # Correlation plots
│   │   └── target.py         # Target-related plots
│   ├── templates/            # Report templates
│   │   ├── default.html      # Default HTML template
│   │   └── default.css       # Default CSS styles
│   ├── __init__.py           # Core module initialization
│   ├── automator.py          # Main EDACore class
│   └── config.py             # Configuration handling
├── unified/                  # Legacy unified module (backward compatibility)
├── requirements/             # Requirement files for dependencies
├── __init__.py               # Package initialization
└── cli.py                    # Command-line interface
```

## Development Guidelines

### Coding Standards

1. **Style Guide**: Follow PEP 8 guidelines for Python code.
2. **Docstrings**: Use Google-style docstrings for all public functions and classes.
3. **Type Hints**: Include type hints for all function parameters and return values.
4. **Line Length**: Keep lines under 100 characters.
5. **Function Size**: Keep functions focused and under 50 lines where possible.

### Testing

1. **Unit Tests**: Write unit tests for all new functions and classes.
2. **Integration Tests**: Add integration tests for end-to-end workflows.
3. **Test Coverage**: Aim for at least 90% test coverage for all modules.
4. **Running Tests**: Use pytest to run the test suite:
   ```bash
   pytest
   ```

### Adding New Functionality

#### Adding a New Analysis Type

1. Create a new module in the `core/analysis/` directory.
2. Implement the analysis function with appropriate parameters.
3. Update the `__init__.py` to expose the function.
4. Add the new analysis to the `EDACore` class in `core/automator.py`.
5. Create corresponding visualization functions in `core/visualization/`.
6. Update report generators to include the new analysis results.

Example of a new analysis module:

```python
"""
Module for analyzing categorical relationships.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

def analyze_categorical_relationships(
    data: pd.DataFrame,
    target_variable: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze relationships between categorical variables.
    
    Args:
        data: DataFrame to analyze
        target_variable: Optional target variable for supervised analysis
        settings: Optional settings dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    results = {}
    # Implementation here
    return results
```

#### Adding a New Visualization

1. Add the visualization function to the appropriate module in `core/visualization/`.
2. Ensure it follows the standard interface and styling guidelines.
3. Update the relevant analysis module to use the new visualization.

Example:

```python
def plot_categorical_relationships(
    data: pd.DataFrame,
    cat_vars: List[str],
    settings: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create a visualization of categorical relationships.
    
    Args:
        data: DataFrame with the data
        cat_vars: List of categorical variables to plot
        settings: Optional settings dictionary
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Implementation here
    return fig
```

#### Adding a New Report Generator

1. Create a new module in `core/report_generators/`.
2. Implement the report generation function.
3. Update the `__init__.py` to expose the function.
4. Add the new report format to the `EDACore.generate_report` method.

#### Extending the Command Line Interface

1. Open `cli.py`.
2. Add a new command or option to the appropriate command group.
3. Implement the corresponding functionality.

### Documentation

1. **Update Docstrings**: Keep docstrings updated with any changes to function signatures.
2. **Developer Documentation**: Update this document when adding significant features.
3. **User Guides**: Update the user guides when adding user-facing features.
4. **Architecture Documentation**: Update architecture docs when changing the system structure.

### Pull Request Process

1. Create a feature branch from `develop`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes, including tests and documentation.

3. Run the full test suite:
   ```bash
   pytest
   ```

4. Update the CHANGELOG.md with your changes.

5. Push your branch and create a pull request against `develop`.

## Performance Considerations

1. **Large Datasets**: Use sampling for large datasets to maintain performance.
2. **Vectorized Operations**: Prefer pandas vectorized operations over loops.
3. **Memory Management**: Consider memory usage when processing large DataFrames.
4. **Caching**: Cache intermediate results for expensive operations.

## Troubleshooting

### Common Development Issues

1. **Import Errors**: Ensure the package is installed in development mode (`pip install -e .`).
2. **Missing Dependencies**: Check that all required packages are installed.
3. **Visualization Issues**: Verify that matplotlib and seaborn are correctly configured.
4. **Test Failures**: Run tests with verbose output to get more details: `pytest -v`.

## Additional Resources

- [Project Roadmap](Roadmap.md): Future development plans
- [Architecture Overview](../architecture/index.md): Detailed architecture documentation
- [Best Practices](CursorBestPractices.md): Coding best practices for the project 