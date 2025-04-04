# EDA Automator

A package to automate Exploratory Data Analysis (EDA) and generate interactive reports in various formats.

## Features

- **Automated analysis** of tabular datasets
- **Interactive visualizations** to explore patterns in the data
- **Multiple report formats**: HTML, Markdown, Excel, and images
- **Command-line interface** for quick analysis
- **Python API** for integration into existing workflows
- **Modular and extensible** for customized analysis

## Installation

```bash
# Basic installation
pip install eda-automator

# With all dependencies for image generation
pip install eda-automator[image]
```

## Quick Usage

### From Command Line

```bash
# Analyze a dataset and generate HTML report
eda-automator analyze data.csv --output reports

# Generate reports in multiple formats
eda-automator analyze data.csv --formats html markdown landscape excel

# Generate a synthetic dataset for testing
eda-automator generate --size 1000 --output test_data.csv
```

### From Python

```python
import pandas as pd
from eda_automator import EDAAutomator, run_analysis

# Load data
df = pd.read_csv('data.csv')

# Simplified method
eda = run_analysis(df, target_variable='target')

# Generate report
eda.generate_html_report('report.html')

# Or use the EDAAutomator class for more control
eda = EDAAutomator(df)
eda.run_basic_analysis()
eda.run_missing_analysis()
eda.run_outlier_analysis()
eda.run_correlation_analysis()
eda.generate_html_report('custom_report.html')
```

## Unified Module

The package includes a unified module that simplifies the analysis process:

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

# Configure environment (optional)
setup_environment(language='en')

# Load or generate data
df = load_data('data.csv')  # or create_dataset(size=1000)

# Analyze the data
eda = run_analysis(df, target_variable='target')

# Generate reports
generate_html_report('report.html', eda)
generate_markdown_report('report.md', eda)
```

## Optional Dependencies

To generate reports in image format, install one of the following combinations:

- `imgkit` + `wkhtmltopdf` (recommended): `pip install imgkit`
- `selenium` + `Chrome webdriver`: `pip install selenium webdriver-manager`
- `weasyprint` + `pdf2image`: `pip install weasyprint pdf2image`

## Contribution

Contributions are welcome. Please open an issue or pull request.

## License

MIT License

## 🌟 Key Features

- **Automated Analysis**: Automatically detects and analyzes important features of your data.
- **High-Quality Visualizations**: Generates informative charts and visualizations for categorical and numerical data.
- **Multi-format Reports**: Creates reports in HTML, Markdown, or image formats.
- **Modular Structure**: Modular architecture for easy extension and customization.
- **Intelligent Detection**: Automatic identification of outliers, relationships, and patterns.

## 🚀 Installation

```bash
pip install eda-automator
```

## 📊 Quick Start

### Modular Approach (Recommended)

```python
from eda_automator import EDAAutomator
import pandas as pd

# Load your data
df = pd.read_csv('my_dataset.csv')

# Create an EDA Automator instance
eda = EDAAutomator(df, target_variable='my_target_variable')

# Run complete analysis
eda.run_full_analysis()

# Generate report
eda.generate_report('eda_report.html', format='html')
```

### Command Line Approach

```bash
# Basic example with synthetic dataset
python examples/unified_eda_modular.py --size 1000 --formats html

# Specify data type and output directory
python examples/unified_eda_modular.py --data-type timeseries --output output/my_report

# View available options
python examples/unified_eda_modular.py --help
```

## 📝 Examples

### Quick Start Jupyter Notebook
Check out the [quickstart notebook](examples/notebooks/quickstart.ipynb) for practical examples.

### Command Line Examples
Try the example scripts in the `examples/` directory:

```bash
python examples/unified_eda_modular.py --formats html markdown --size 1000
```

## 🧩 Project Structure

```
eda_automator/                 # Main package
├── unified/                   # Centralized unified module
│   ├── report_generators/     # Unified report generators
│   ├── __init__.py            # Unified module initialization
│   ├── analysis.py            # Data analysis
│   ├── config.py              # Configuration
│   ├── data.py                # Data handling
│   ├── dependencies.py        # Dependency verification
│   ├── main.py                # Main functionality
│   ├── utils.py               # Utilities
│   └── visualizations.py      # Visualizations
├── report_generators/         # Classic report generators
├── __init__.py                # Package initialization
├── bivariate.py               # Bivariate analysis
├── cli.py                     # Command line interface
├── data_quality.py            # Data quality analysis
├── multivariate.py            # Multivariate analysis
├── report.py                  # Report generation
├── stats_analysis.py          # Statistical analysis
├── univariate.py              # Univariate analysis
├── utils.py                   # Utility functions
└── visuals.py                 # Visualizations

examples/                      # Usage examples
├── notebooks/                 # Example notebooks
├── unified_eda_modular.py     # Example script for unified module
└── README.md                  # Examples documentation
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- To all contributors and users of EDA Automator.
- To the data science community for their continued inspiration and support. 