#!/usr/bin/env python
"""
Command Line Interface (CLI) for EDA Automator

This module provides a command-line interface for running
exploratory data analysis tasks.
"""

import argparse
import os
import sys
import logging
import datetime
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple

# Import from core modules
from eda_automator.core import EDACore
from eda_automator.core.data import load_data, create_dataset
from eda_automator.core.utils import setup_logging, setup_environment
from eda_automator.core.report_generators import generate_report

# Set up logger
logger = logging.getLogger('eda_automator')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='EDA Automator - Automated Exploratory Data Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a dataset')
    analyze_parser.add_argument('input', help='Input file path (CSV, Excel, Parquet, etc.)')
    analyze_parser.add_argument('-o', '--output', help='Output directory for reports', default='./eda_reports')
    analyze_parser.add_argument('-f', '--format', help='Report format', 
                              choices=['html', 'markdown', 'md', 'image'], default='html')
    analyze_parser.add_argument('-t', '--target', help='Target variable for supervised analysis')
    analyze_parser.add_argument('--outlier-method', help='Outlier detection method',
                              choices=['z-score', 'iqr', 'isolation-forest'], default='z-score')
    analyze_parser.add_argument('--correlation-method', help='Correlation method',
                              choices=['pearson', 'spearman', 'kendall'], default='pearson')
    analyze_parser.add_argument('--no-plot', help='Disable plot generation', action='store_true')
    analyze_parser.add_argument('--no-code', help='Disable code snippet generation', action='store_true')
    analyze_parser.add_argument('--sample', help='Sample size for large datasets', type=int)
    
    # Generate dataset command
    dataset_parser = subparsers.add_parser('dataset', help='Generate a synthetic dataset')
    dataset_parser.add_argument('-o', '--output', help='Output file path', required=True)
    dataset_parser.add_argument('-s', '--size', help='Number of rows', type=int, default=1000)
    dataset_parser.add_argument('-c', '--columns', help='Number of columns', type=int, default=10)
    dataset_parser.add_argument('-t', '--type', help='Dataset type',
                              choices=['basic', 'timeseries', 'classification', 'regression'], 
                              default='basic')
    dataset_parser.add_argument('--seed', help='Random seed', type=int, default=42)
    
    return parser.parse_args()

def main():
    """Main CLI entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    
    # Set up environment
    env_settings = setup_environment()
    
    if args.command == 'analyze':
        logger.info(f"Analyzing dataset: {args.input}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)
        
        try:
            # Load data
            logger.info("Loading data...")
            df = load_data(args.input)
            
            # Initialize EDA Core
            settings = {
                'sampling_threshold': args.sample if args.sample else env_settings.get('sampling_threshold', 10000)
            }
            
            eda = EDACore(dataframe=df, target_variable=args.target, settings=settings)
            
            # Run analysis
            logger.info("Running analysis...")
            results = eda.run_full_analysis(
                outlier_method=args.outlier_method,
                correlation_method=args.correlation_method
            )
            
            # Generate figures if not disabled
            figures = None
            if not args.no_plot:
                logger.info("Generating visualizations...")
                figures = eda.generate_visualizations()
            
            # Generate report
            logger.info(f"Generating {args.format} report...")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eda_report_{timestamp}"
            
            output_path = os.path.join(args.output, f"{filename}.{args.format}")
            if args.format == 'md':
                output_path = os.path.join(args.output, f"{filename}.md")
            elif args.format == 'markdown':
                output_path = os.path.join(args.output, f"{filename}.md")
            elif args.format == 'image':
                output_path = os.path.join(args.output, f"{filename}.png")
            
            report_path = generate_report(
                output_path=output_path,
                format=args.format,
                data=df,
                results=results,
                figures=figures,
                target_variable=args.target,
                include_code=not args.no_code
            )
            
            logger.info(f"Report generated: {report_path}")
            print(f"\nReport generated: {os.path.abspath(report_path)}")
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return 1
    
    elif args.command == 'dataset':
        logger.info(f"Generating synthetic dataset: {args.type}")
        
        try:
            # Generate dataset
            df = create_dataset(
                dataset_type=args.type,
                n_samples=args.size,
                n_features=args.columns,
                random_state=args.seed
            )
            
            # Save dataset
            _, ext = os.path.splitext(args.output)
            if not ext:
                # Default to CSV if no extension
                args.output = args.output + '.csv'
                ext = '.csv'
            
            # Save in appropriate format
            if ext.lower() == '.csv':
                df.to_csv(args.output, index=False)
            elif ext.lower() in ['.xlsx', '.xls']:
                df.to_excel(args.output, index=False)
            elif ext.lower() == '.parquet':
                df.to_parquet(args.output, index=False)
            elif ext.lower() in ['.json', '.js']:
                df.to_json(args.output, orient='records')
            elif ext.lower() in ['.pkl', '.pickle']:
                df.to_pickle(args.output)
            else:
                logger.warning(f"Unrecognized extension: {ext}, defaulting to CSV")
                df.to_csv(args.output, index=False)
            
            logger.info(f"Dataset generated: {args.output}")
            print(f"\nDataset generated: {os.path.abspath(args.output)}")
            
        except Exception as e:
            logger.error(f"Error generating dataset: {str(e)}")
            return 1
    
    else:
        # No command specified, show help
        print("No command specified. Use --help for usage information.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 