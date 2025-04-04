"""
Tests for the EDAAutomator class.
"""

import pytest
import pandas as pd
import numpy as np
from eda_automator import EDAAutomator


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    
    # Create a DataFrame with different variable types
    df = pd.DataFrame({
        'numeric1': np.random.normal(10, 2, 100),
        'numeric2': np.random.uniform(0, 100, 100),
        'categorical1': np.random.choice(['A', 'B', 'C'], 100),
        'categorical2': np.random.choice(['X', 'Y', 'Z'], 100),
        'boolean': np.random.choice([True, False], 100),
        'datetime': pd.date_range(start='2020-01-01', periods=100, freq='D'),
        'with_missing': np.random.normal(5, 1, 100)
    })
    
    # Add some missing values
    df.loc[np.random.choice(len(df), 10), 'with_missing'] = np.nan
    
    return df


def test_eda_automator_initialization(sample_dataframe):
    """Test that EDAAutomator initializes correctly."""
    # Initialize with default parameters
    eda = EDAAutomator(sample_dataframe)
    
    # Check that the DataFrame was stored correctly
    assert eda.df.equals(sample_dataframe)
    assert eda.target is None
    
    # Initialize with a target variable
    eda = EDAAutomator(sample_dataframe, target_variable='numeric1')
    assert eda.target == 'numeric1'
    
    # Check that variable types were identified
    assert isinstance(eda.var_types, dict)
    assert 'numerical' in eda.var_types
    assert 'categorical' in eda.var_types


def test_eda_automator_data_quality_analysis(sample_dataframe):
    """Test the data quality analysis functionality."""
    eda = EDAAutomator(sample_dataframe)
    
    # Run data quality analysis
    quality_results = eda.run_data_quality_analysis()
    
    # Check that basic results are returned
    assert isinstance(quality_results, dict)
    assert 'quality_score' in quality_results
    assert 'missing_values' in quality_results
    assert 'outliers' in quality_results
    
    # Check that results are stored in the object
    assert 'data_quality' in eda.results
    assert eda.results['data_quality'] == quality_results


def test_eda_automator_statistical_analysis(sample_dataframe):
    """Test the statistical analysis functionality."""
    eda = EDAAutomator(sample_dataframe)
    
    # Run statistical analysis
    stats_results = eda.run_statistical_analysis()
    
    # Check that basic results are returned
    assert isinstance(stats_results, dict)
    
    # Check that results are stored in the object
    assert 'statistics' in eda.results
    assert eda.results['statistics'] == stats_results


def test_eda_automator_univariate_analysis(sample_dataframe):
    """Test the univariate analysis functionality."""
    eda = EDAAutomator(sample_dataframe)
    
    # Run univariate analysis
    univariate_results = eda.run_univariate_analysis()
    
    # Check that basic results are returned
    assert isinstance(univariate_results, dict)
    assert 'plots' in univariate_results
    
    # Check that results are stored in the object
    assert 'univariate' in eda.results
    assert eda.results['univariate'] == univariate_results


def test_eda_automator_bivariate_analysis(sample_dataframe):
    """Test the bivariate analysis functionality."""
    eda = EDAAutomator(sample_dataframe)
    
    # Run bivariate analysis
    bivariate_results = eda.run_bivariate_analysis()
    
    # Check that basic results are returned
    assert isinstance(bivariate_results, dict)
    assert 'plots' in bivariate_results
    
    # Check that results are stored in the object
    assert 'bivariate' in eda.results
    assert eda.results['bivariate'] == bivariate_results


def test_eda_automator_multivariate_analysis(sample_dataframe):
    """Test the multivariate analysis functionality."""
    eda = EDAAutomator(sample_dataframe)
    
    # Run multivariate analysis
    multivariate_results = eda.run_multivariate_analysis()
    
    # Check that basic results are returned
    assert isinstance(multivariate_results, dict)
    
    # Check that results are stored in the object
    assert 'multivariate' in eda.results
    assert eda.results['multivariate'] == multivariate_results


def test_eda_automator_full_analysis(sample_dataframe):
    """Test the full analysis functionality."""
    eda = EDAAutomator(sample_dataframe)
    
    # Run full analysis without generating a report
    results = eda.run_full_analysis()
    
    # Check that all analysis sections are present
    assert isinstance(results, dict)
    assert 'data_quality' in results
    assert 'statistics' in results
    assert 'univariate' in results
    assert 'bivariate' in results
    assert 'multivariate' in results


def test_eda_automator_report_generation(sample_dataframe, tmp_path):
    """Test report generation functionality."""
    eda = EDAAutomator(sample_dataframe)
    
    # Run a full analysis
    eda.run_full_analysis()
    
    # Generate a report without saving it
    report_content = eda.generate_report()
    assert isinstance(report_content, str)
    assert len(report_content) > 0
    
    # Generate and save an HTML report
    report_path = tmp_path / "test_report.html"
    saved_path = eda.generate_report(output_path=str(report_path))
    assert saved_path == str(report_path)
    assert report_path.exists()
    
    # Generate a Markdown report
    md_report_path = tmp_path / "test_report.md"
    saved_md_path = eda.generate_report(output_path=str(md_report_path), format='markdown')
    assert saved_md_path == str(md_report_path)
    assert md_report_path.exists() 