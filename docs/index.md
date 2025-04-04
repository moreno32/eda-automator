# EDA Automator Documentation

Welcome to the EDA Automator documentation. This comprehensive guide covers everything you need to know about the EDA Automator library, from installation and basic usage to advanced features and development.

## About EDA Automator

EDA Automator is a Python library designed to automate and standardize the exploratory data analysis process. It provides advanced tools for:

- **Modular Analysis**: 
  - Comprehensive architecture with dedicated modules for different analysis types
  - Organized in a clean, maintainable directory structure
  - Independent components that can be used separately or together

- **Data Quality Analysis**: 
  - Comprehensive missing values detection and visualization
  - Outlier identification using multiple methods (Z-score, IQR, isolation forest)
  - Data type consistency checks
  - Duplicate record analysis

- **Statistical Analysis**: 
  - Robust descriptive statistics
  - Distribution tests and normality analysis
  - Advanced correlation analysis
  - Statistical significance testing

- **Visualization**: 
  - Professional-grade plots with consistent styling
  - Interactive visualizations in HTML reports
  - Automatic handling of large datasets through sampling
  - Customizable color palettes and themes

- **Reports**: 
  - Comprehensive HTML and Markdown reports
  - Image reports for presentations
  - Code snippets for reproducibility
  - Customizable report sections

## New Architecture

EDA Automator has been refactored with a modular architecture:

```
eda_automator/
├── core/                     # Main module
│   ├── analysis/             # Data analysis modules
│   ├── data/                 # Data loading and generation
│   ├── report_generators/    # Report generators (HTML, Markdown, Image)
│   ├── utils/                # Common utilities
│   ├── visualization/        # Visualizations
│   ├── templates/            # Report templates
│   ├── automator.py          # Main EDACore class
│   └── config.py             # Configuration
├── unified/                  # Unified module (backward compatibility)
├── requirements/             # Requirements files
└── cli.py                    # Command-line interface
```

This new structure improves maintainability, testability, and extensibility while providing a cleaner API.

## Documentation Sections

- [**User Guides**](user_guides/index.md): Step-by-step guides and tutorials
- [**Architecture**](architecture/index.md): System design and components
- [**Development**](development/index.md): Contribution guidelines and API reference

## Quick Links

- [Quick Start Guide](user_guides/quick_start.md)
- [Configuration Options](user_guides/configuration.md)
- [Project Roadmap](Roadmap.md)
- [API Reference](development/api_reference.md)

## Installation

```bash
pip install eda-automator
```

For image report generation support:

```bash
pip install -r eda_automator/requirements/image_generation.txt
```

## Basic Usage

```python
from eda_automator.core import EDACore
import pandas as pd

# Load the dataset
data = pd.read_csv("your_dataset.csv")

# Initialize EDA Automator with custom configuration
eda = EDACore(
    dataframe=data,
    target_variable="target_column",  # Optional target variable
    settings={
        "sampling_threshold": 10000,  # Automatic sampling for large datasets
        "outlier_method": "z-score",
        "correlation_method": "pearson"
    }
)

# Run comprehensive analysis
results = eda.run_full_analysis()

# Generate visualizations
figures = eda.generate_visualizations()

# Generate a report in different formats
eda.generate_report(
    output_path="eda_report.html",
    format="html",
    include_code=True
)

# Generate reports in different formats
eda.generate_report(output_path="eda_report.md", format="markdown")
eda.generate_report(output_path="eda_report.png", format="image")
```

## Using the Command Line Interface

```bash
# Analyze a dataset
eda-automator analyze data.csv -o reports -f html -t target_column

# Generate a synthetic dataset
eda-automator dataset -o generated_data.csv -s 1000 -c 10 -t basic
```

## Key Features

- **Modular Architecture**:
  - Clean separation of concerns for different analysis types
  - Independent components that work well together
  - Reusable modules for custom analysis workflows

- **Enhanced Visualizations**:
  - Modern and accessible color palettes
  - Professional styling with customizable themes
  - Intelligent handling of outliers and missing values
  - Automatic sampling for large datasets

- **Flexible Report Generation**:
  - Multiple formats: HTML, Markdown, and Image
  - Customizable templates and styling
  - Interactive elements in HTML reports
  - Code snippets for reproducibility

- **Smart Data Handling**:
  - Automatic data type detection and appropriate analysis
  - Intelligent sampling strategies for large datasets
  - Missing value pattern analysis
  - Duplicate detection and handling

## Getting Help

If you need help with EDA Automator, you can:

- Check the [Frequently Asked Questions](user_guides/faq.md)
- Review [Common Issues](user_guides/troubleshooting.md)
- [Report a Bug](https://github.com/yourusername/eda_automator/issues)
- Join our [Community Discussion](https://github.com/yourusername/eda_automator/discussions)

## License

EDA Automator is available under the MIT License. See the [LICENSE](https://github.com/yourusername/eda_automator/LICENSE) file for more details. 