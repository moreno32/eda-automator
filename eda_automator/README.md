# EDA Automator

Tool for automating Exploratory Data Analysis (EDA).

## Project Structure

The project architecture has been reorganized into a modular structure:

```
eda_automator/
├── core/                     # Main module
│   ├── analysis/             # Data analysis
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

## Installation

```bash
pip install eda-automator
```

Or to install from the repository:

```bash
git clone https://github.com/yourusername/eda-automator.git
cd eda-automator
pip install -e .
```

## Basic Usage

### From Python

```python
# Import EDACore
from eda_automator.core import EDACore
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Create EDACore instance
eda = EDACore(dataframe=df, target_variable="target")

# Run analysis
results = eda.run_full_analysis()

# Generate visualizations
figures = eda.generate_visualizations()

# Generate report
eda.generate_report(output_path="report.html", format="html")
```

### From the Command Line

```bash
# Analyze a dataset
eda-automator analyze data.csv -o reports -f html -t target

# Generate a synthetic dataset
eda-automator dataset -o generated_data.csv -s 1000 -c 10 -t basic
```

## Report Generation

EDA Automator supports multiple report formats:

- HTML (`format="html"`)
- Markdown (`format="markdown"` or `format="md"`)
- Image (`format="image"`)

### Example:

```python
# Generate HTML report
eda.generate_report(output_path="report.html", format="html")

# Generate Markdown report
eda.generate_report(output_path="report.md", format="markdown")

# Generate report as an image
eda.generate_report(output_path="report.png", format="image")
```

## Visualizations

The visualization module automatically generates charts for different types of analysis:

- Basic dataset information
- Missing values analysis
- Outlier detection
- Correlation analysis
- Distribution analysis
- Target variable analysis

## Extensibility

The modular design makes it easy to extend the package's capabilities:

- Add new types of analysis in `core/analysis/`
- Create new report generators in `core/report_generators/`
- Implement new visualizations in `core/visualization/`

## License

MIT 