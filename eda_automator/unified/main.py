"""
Unified EDA Reports - Main Module

This is the main entry point for the unified EDA reports.
It orchestrates the data loading, analysis, visualization, and report generation.
"""

import os
import sys
import pandas as pd
import argparse
import time
import warnings

# Add the parent directory to the path so we can import the eda_automator package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from eda_automator import EDAAutomator
from eda_automator.report_generators import generate_alternative_html, generate_excel_report, fix_html_template

from .config import (
    DEFAULT_OUTPUT_DIR, 
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_MISSING_THRESHOLD,
    DEFAULT_CORRELATION_THRESHOLD
)
from .data import load_data, validate_data
from .analysis import (
    perform_basic_analysis,
    analyze_missing_data,
    analyze_outliers,
    analyze_correlations,
    perform_target_analysis
)
from .visualizations import (
    create_data_overview_visualizations,
    create_categorical_visualizations,
    create_numerical_visualizations,
    create_correlation_heatmap
)
from .report_generators.html_generator import generate_html_report

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Unified EDA Reports')
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the input data file (CSV, Excel, or Parquet)')
    
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for reports (default: {DEFAULT_OUTPUT_DIR})')
    
    parser.add_argument('--target', type=str, default=None,
                        help='Target variable name for supervised analysis')
    
    parser.add_argument('--format', type=str, choices=['html', 'excel', 'all'], 
                        default='html',
                        help='Report format (default: html)')
    
    parser.add_argument('--quality-threshold', type=float, 
                        default=DEFAULT_QUALITY_THRESHOLD,
                        help=f'Threshold for data quality alerts (default: {DEFAULT_QUALITY_THRESHOLD})')
    
    parser.add_argument('--missing-threshold', type=float, 
                        default=DEFAULT_MISSING_THRESHOLD,
                        help=f'Threshold for missing data alerts (default: {DEFAULT_MISSING_THRESHOLD})')
    
    parser.add_argument('--correlation-threshold', type=float, 
                        default=DEFAULT_CORRELATION_THRESHOLD,
                        help=f'Threshold for correlation alerts (default: {DEFAULT_CORRELATION_THRESHOLD})')
    
    parser.add_argument('--generate-excel', action='store_true',
                        help='Generate Excel report')
    
    parser.add_argument('--suppress-warnings', action='store_true',
                        help='Suppress warning messages')
    
    return parser.parse_args()

def main():
    """Main function to run the unified EDA reports."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Suppress warnings if requested
    if args.suppress_warnings:
        warnings.filterwarnings('ignore')
    
    # Start timer
    start_time = time.time()
    
    print(f"Starting Unified EDA Reports for {args.data}")
    
    try:
        # Load data
        df = load_data(args.data)
        
        # Validate data
        validate_data(df)
        
        # Set the output directory
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        
        # Create EDA Automator instance
        eda = EDAAutomator(df)
        
        # Set target variable if provided
        if args.target and args.target in df.columns:
            eda.set_target(args.target)
            print(f"Target variable set to: {args.target}")
        
        # Run analysis
        print("\nRunning analysis...")
        eda.analyze_data_quality()
        eda.analyze_missing_data()
        eda.analyze_outliers()
        eda.analyze_correlations()
        
        # Generate reports based on format
        print("\nGenerating reports...")
        
        # Generate HTML report
        if args.format in ['html', 'all']:
            html_path = os.path.join(output_dir, 'eda_report.html')
            generate_alternative_html(html_path, eda)
        
        # Generate Excel report
        if args.format in ['excel', 'all'] or args.generate_excel:
            excel_path = os.path.join(output_dir, 'eda_report.xlsx')
            generate_excel_report(excel_path, eda)
        
        elapsed_time = time.time() - start_time
        print(f"\nAnalysis completed in {elapsed_time:.2f} seconds.")
        print(f"Reports saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 