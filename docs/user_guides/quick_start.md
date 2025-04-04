# Quick Start Guide

This guide will help you get started with EDA Automator quickly and efficiently.

## Installation

Install EDA Automator using pip:

```bash
pip install eda-automator
```

## Basic Usage

Here's how to perform a basic exploratory data analysis:

```python
import pandas as pd
from eda_automator import EDAAutomator

# Load your dataset
data = pd.read_csv("your_dataset.csv")

# Initialize EDA Automator
automator = EDAAutomator(data, target_variable="target_column")

# Run a complete analysis and generate a report
automator.run_full_analysis()
automator.generate_report("eda_report.html")
```

## Step-by-Step Analysis

If you prefer to run individual analysis components:

```python
# Initialize
automator = EDAAutomator(df)

# 1. Data Quality Analysis
quality_results = automator.run_data_quality_analysis()
print(f"Data quality score: {quality_results['quality_score']}")

# 2. Statistical Analysis
stats_results = automator.run_statistical_analysis()

# 3. Univariate Analysis
univariate_results = automator.run_univariate_analysis()

# 4. Bivariate Analysis
bivariate_results = automator.run_bivariate_analysis()

# 5. Multivariate Analysis
multivariate_results = automator.run_multivariate_analysis()

# 6. Generate Report
automator.generate_report("step_by_step_report.html")
```

## Configuration Options

Customize the analysis with configuration parameters:

```python
# Custom configuration
config = {
    'missing_threshold': 0.1,         # Mark columns with >10% missing values
    'outlier_method': 'iqr',          # Use IQR for outlier detection
    'sampling_threshold': 5000,       # Sample data if rows exceed 5000
    'correlation_method': 'spearman', # Use Spearman correlation
    'palette_type': 'categorical'     # Use categorical color palette
}

# Initialize with custom configuration
automator = EDAAutomator(data, **config)
```

## Sample Dataset

If you don't have a dataset ready, EDA Automator includes sample data generation:

```python
from eda_automator.utils import create_sample_dataset

# Generate a sample dataset with 1000 rows
sample_data = create_sample_dataset(size=1000)

# Run analysis on sample data
automator = EDAAutomator(sample_data)
automator.run_full_analysis()
```

## Next Steps

- Learn about [detailed configuration options](configuration.md)
- See [data quality assessment techniques](data_quality.md)
- Explore [visualization capabilities](visualization.md)
- Check out [report generation best practices](report_best_practices.md)

For more examples, see the [examples directory](https://github.com/yourusername/eda_automator/tree/main/examples) in the repository. 