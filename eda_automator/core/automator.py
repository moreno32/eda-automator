"""
EDA Automator Core Class

This module provides the main EDACore class that orchestrates
exploratory data analysis operations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
import os
import time
import warnings

from eda_automator.core.utils import get_logger, setup_environment
from eda_automator.core.utils.dependencies import check_dependencies, get_missing_dependencies

class EDACore:
    """
    Core class for exploratory data analysis.
    
    This class orchestrates the analysis process, providing a convenient
    interface for loading data, running analyses, generating visualizations,
    and creating reports.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame, optional
        Input DataFrame for analysis. If None, can be loaded later
    target_variable : str, optional
        Name of the target variable for supervised analysis
    settings : dict, optional
        Configuration settings
    """
    
    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        target_variable: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ):
        # Initialize logger
        self.logger = get_logger()
        self.logger.info("Initializing EDA Automator")
        
        # Apply environmental settings
        self.settings = setup_environment(**(settings or {}))
        
        # Initialize data attributes
        self.df = dataframe
        self.target_variable = target_variable
        self.original_df = None if dataframe is None else dataframe.copy()
        
        # Results containers
        self.results = {
            'basic': {},
            'missing_values': {},
            'outliers': {},
            'correlation': {},
            'distribution': {},
            'target_analysis': {}
        }
        
        # Visualization containers
        self.figures = {
            'basic': {},
            'missing_values': {},
            'outliers': {},
            'correlation': {},
            'distribution': {},
            'target_analysis': {}
        }
        
        # Analysis status tracking
        self.analysis_status = {
            'basic': False,
            'missing_values': False,
            'outliers': False,
            'correlation': False,
            'distribution': False,
            'target_analysis': False
        }
        
        # Check dependencies
        self._check_required_dependencies()
        
        # If data is provided, start with basic analysis
        if dataframe is not None:
            self._validate_dataframe()
            
            # Apply sampling if dataframe is large
            self._apply_sampling()
            
            # Perform basic analysis
            self.run_basic_analysis()
    
    def _check_required_dependencies(self) -> None:
        """Check for required dependencies and log warnings for missing ones."""
        required_deps = ['pandas', 'numpy', 'matplotlib']
        missing = get_missing_dependencies(required_deps)
        
        if missing:
            warning_msg = f"Missing required dependencies: {', '.join(missing)}"
            self.logger.warning(warning_msg)
            warnings.warn(warning_msg)
    
    def _validate_dataframe(self) -> None:
        """Validate the input DataFrame."""
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Validate target variable if specified
        if self.target_variable is not None:
            if self.target_variable not in self.df.columns:
                raise ValueError(f"Target variable '{self.target_variable}' not found in DataFrame")
                
        self.logger.info(f"Validated DataFrame with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
    
    def _apply_sampling(self) -> None:
        """Apply sampling if DataFrame exceeds threshold size."""
        threshold = self.settings.get('sampling_threshold', 10000)
        
        if len(self.df) > threshold:
            self.logger.info(f"DataFrame exceeds sampling threshold ({threshold} rows). Applying sampling.")
            sample_size = threshold
            
            # Keep original DataFrame
            if self.original_df is None:
                self.original_df = self.df.copy()
                
            # Sample dataframe, keeping distribution of target if available
            if self.target_variable is not None and self.target_variable in self.df.columns:
                # Stratified sampling for categorical target
                if pd.api.types.is_categorical_dtype(self.df[self.target_variable]) or \
                   len(self.df[self.target_variable].unique()) <= 10:
                    self.df = self.df.groupby(self.target_variable, group_keys=False).apply(
                        lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(self.df))))
                    )
                else:
                    # Random sampling for continuous target
                    self.df = self.df.sample(sample_size)
            else:
                # Random sampling
                self.df = self.df.sample(sample_size)
                
            self.logger.info(f"Sampled DataFrame to {len(self.df)} rows")
    
    def load_data(self, 
                  data: Union[str, pd.DataFrame], 
                  target_variable: Optional[str] = None,
                  **kwargs) -> None:
        """
        Load data from a file or DataFrame.
        
        Parameters
        ----------
        data : str or pandas.DataFrame
            Path to data file or DataFrame
        target_variable : str, optional
            Name of the target variable for supervised analysis
        **kwargs : dict
            Additional keyword arguments to pass to pandas read functions
        """
        start_time = time.time()
        self.logger.info("Loading data")
        
        if isinstance(data, pd.DataFrame):
            self.df = data
            self.original_df = data.copy()
        elif isinstance(data, str):
            # Determine file type and load accordingly
            file_ext = os.path.splitext(data.lower())[1]
            
            try:
                if file_ext == '.csv':
                    self.df = pd.read_csv(data, **kwargs)
                elif file_ext in ['.xlsx', '.xls']:
                    self.df = pd.read_excel(data, **kwargs)
                elif file_ext == '.parquet':
                    self.df = pd.read_parquet(data, **kwargs)
                elif file_ext in ['.json', '.js']:
                    self.df = pd.read_json(data, **kwargs)
                elif file_ext in ['.pkl', '.pickle']:
                    self.df = pd.read_pickle(data, **kwargs)
                elif file_ext == '.feather':
                    self.df = pd.read_feather(data, **kwargs)
                else:
                    raise ValueError(f"Unsupported file extension: {file_ext}")
                
                self.original_df = self.df.copy()
            except Exception as e:
                self.logger.error(f"Error loading data: {str(e)}")
                raise
        else:
            raise TypeError("Data must be a pandas DataFrame or a file path")
        
        # Set target variable if provided
        if target_variable is not None:
            self.target_variable = target_variable
        
        # Validate and sample dataframe
        self._validate_dataframe()
        self._apply_sampling()
        
        # Perform basic analysis
        self.run_basic_analysis()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Data loaded and analyzed in {elapsed_time:.2f} seconds")
    
    def run_basic_analysis(self) -> Dict[str, Any]:
        """
        Run basic analysis on the DataFrame.
        
        This includes data types, basic statistics, and data preview.
        
        Returns
        -------
        dict
            Dictionary of basic analysis results
        """
        self.logger.info("Running basic analysis")
        
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() method first.")
        
        # Get basic information
        n_rows, n_cols = self.df.shape
        dtypes = self.df.dtypes.astype(str).to_dict()
        
        # Identify data types
        numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=['category', 'object']).columns.tolist()
        datetime_columns = self.df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        boolean_columns = self.df.select_dtypes(include=['bool']).columns.tolist()
        
        # Basic stats for numeric columns
        numeric_stats = {}
        if numeric_columns:
            numeric_stats = self.df[numeric_columns].describe().to_dict()
        
        # Category counts for categorical columns
        category_counts = {}
        for col in categorical_columns:
            value_counts = self.df[col].value_counts().head(10).to_dict()
            category_counts[col] = value_counts
        
        # Missing values summary
        missing_counts = self.df.isna().sum().to_dict()
        missing_percentages = (self.df.isna().mean() * 100).to_dict()
        
        # Store results
        self.results['basic'] = {
            'shape': {'rows': n_rows, 'columns': n_cols},
            'dtypes': dtypes,
            'column_types': {
                'numeric': numeric_columns,
                'categorical': categorical_columns,
                'datetime': datetime_columns,
                'boolean': boolean_columns
            },
            'numeric_stats': numeric_stats,
            'category_counts': category_counts,
            'missing_counts': missing_counts,
            'missing_percentages': missing_percentages,
            'memory_usage': self.df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            'preview': self.df.head(5).to_dict('records')
        }
        
        self.analysis_status['basic'] = True
        self.logger.info("Basic analysis completed")
        
        return self.results['basic']
    
    def run_missing_analysis(self) -> Dict[str, Any]:
        """
        Run missing values analysis.
        
        Returns
        -------
        dict
            Dictionary of missing values analysis results
        """
        from eda_automator.core.analysis import analyze_missing_values
        
        self.logger.info("Running missing values analysis")
        
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() method first.")
        
        # Run missing values analysis
        self.results['missing_values'] = analyze_missing_values(self.df)
        self.analysis_status['missing_values'] = True
        
        return self.results['missing_values']
    
    def run_outlier_analysis(self, method: str = 'z-score', threshold: float = 3.0) -> Dict[str, Any]:
        """
        Run outlier detection analysis.
        
        Parameters
        ----------
        method : str, default 'z-score'
            Method to use for outlier detection ('z-score' or 'iqr')
        threshold : float, default 3.0
            Threshold for outlier detection
            
        Returns
        -------
        dict
            Dictionary of outlier analysis results
        """
        from eda_automator.core.analysis import detect_outliers
        
        self.logger.info(f"Running outlier analysis with method={method}, threshold={threshold}")
        
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() method first.")
        
        # Run outlier detection
        self.results['outliers'] = detect_outliers(
            self.df,
            method=method,
            threshold=threshold
        )
        self.analysis_status['outliers'] = True
        
        return self.results['outliers']
    
    def run_correlation_analysis(self, method: str = 'pearson', threshold: float = 0.7) -> Dict[str, Any]:
        """
        Run correlation analysis.
        
        Parameters
        ----------
        method : str, default 'pearson'
            Correlation method ('pearson', 'spearman', or 'kendall')
        threshold : float, default 0.7
            Threshold for significant correlations
            
        Returns
        -------
        dict
            Dictionary of correlation analysis results
        """
        from eda_automator.core.analysis import correlate_features
        
        self.logger.info(f"Running correlation analysis with method={method}, threshold={threshold}")
        
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() method first.")
        
        # Run correlation analysis
        self.results['correlation'] = correlate_features(
            self.df,
            method=method,
            threshold=threshold
        )
        self.analysis_status['correlation'] = True
        
        return self.results['correlation']
    
    def run_distribution_analysis(self) -> Dict[str, Any]:
        """
        Run distribution analysis.
        
        Returns
        -------
        dict
            Dictionary of distribution analysis results
        """
        from eda_automator.core.analysis import analyze_distributions
        
        self.logger.info("Running distribution analysis")
        
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() method first.")
        
        # Run distribution analysis
        self.results['distribution'] = analyze_distributions(self.df)
        self.analysis_status['distribution'] = True
        
        return self.results['distribution']
    
    def run_target_analysis(self) -> Dict[str, Any]:
        """
        Run target variable analysis.
        
        Returns
        -------
        dict
            Dictionary of target analysis results
        """
        from eda_automator.core.analysis import analyze_target_relationship
        
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() method first.")
            
        if self.target_variable is None:
            self.logger.warning("No target variable specified. Skipping target analysis.")
            return {}
        
        self.logger.info(f"Running target analysis for '{self.target_variable}'")
        
        # Run target analysis
        self.results['target_analysis'] = analyze_target_relationship(
            self.df,
            self.target_variable
        )
        self.analysis_status['target_analysis'] = True
        
        return self.results['target_analysis']
    
    def run_full_analysis(self, 
                          outlier_method: str = 'z-score',
                          outlier_threshold: float = 3.0,
                          correlation_method: str = 'pearson',
                          correlation_threshold: float = 0.7) -> Dict[str, Dict[str, Any]]:
        """
        Run all available analyses.
        
        Parameters
        ----------
        outlier_method : str, default 'z-score'
            Method to use for outlier detection
        outlier_threshold : float, default 3.0
            Threshold for outlier detection
        correlation_method : str, default 'pearson'
            Correlation method
        correlation_threshold : float, default 0.7
            Threshold for significant correlations
            
        Returns
        -------
        dict
            Dictionary of all analysis results
        """
        self.logger.info("Running full analysis")
        
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() method first.")
        
        # Run all analyses
        self.run_basic_analysis()
        self.run_missing_analysis()
        self.run_outlier_analysis(method=outlier_method, threshold=outlier_threshold)
        self.run_correlation_analysis(method=correlation_method, threshold=correlation_threshold)
        self.run_distribution_analysis()
        
        # Run target analysis if target variable is specified
        if self.target_variable is not None:
            self.run_target_analysis()
        
        self.logger.info("Full analysis completed")
        
        return self.results
    
    def generate_visualizations(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate visualizations for analysis results.
        
        Parameters
        ----------
        category : str, optional
            Specific category of visualizations to generate.
            If None, generates all applicable visualizations
            
        Returns
        -------
        dict
            Dictionary of generated figures
        """
        from eda_automator.core.visualization import generate_figures
        
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() method first.")
        
        # Determine which categories to generate
        categories = []
        if category is not None:
            if category not in self.analysis_status:
                raise ValueError(f"Invalid category: {category}")
            categories = [category]
        else:
            categories = [cat for cat, status in self.analysis_status.items() if status]
        
        self.logger.info(f"Generating visualizations for categories: {categories}")
        
        # Generate figures for each category
        for cat in categories:
            if self.analysis_status[cat]:
                self.figures[cat] = generate_figures(
                    self.df,
                    cat,
                    self.results[cat],
                    target_variable=self.target_variable
                )
        
        self.logger.info("Visualization generation completed")
        
        return self.figures
    
    def generate_report(self, output_path: str, report_format: str = 'html', **kwargs) -> str:
        """
        Generate a report of the analysis results.
        
        Parameters
        ----------
        output_path : str
            Path to save the report
        report_format : str, default 'html'
            Format of the report ('html', 'markdown', 'excel', 'landscape', 'portrait')
        **kwargs : dict
            Additional keyword arguments to pass to the report generator
            
        Returns
        -------
        str
            Path to the generated report
        """
        from eda_automator.core.report_generators import generate_report
        
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() method first.")
        
        # Run any analyses that haven't been run yet
        if not self.analysis_status['basic']:
            self.run_basic_analysis()
        
        # Generate any visualizations that haven't been generated yet
        for cat, status in self.analysis_status.items():
            if status and not self.figures.get(cat):
                self.generate_visualizations(cat)
        
        self.logger.info(f"Generating {report_format} report to {output_path}")
        
        # Generate report
        report_path = generate_report(
            output_path=output_path,
            format=report_format,
            data=self.df,
            results=self.results,
            figures=self.figures,
            target_variable=self.target_variable,
            settings=self.settings,
            **kwargs
        )
        
        self.logger.info(f"Report generated: {report_path}")
        
        return report_path 