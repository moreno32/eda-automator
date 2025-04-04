"""
Data loading utilities for EDA Automator

This module provides functions for loading data from various sources.
"""

import pandas as pd
import os
import urllib.request
from typing import Optional, Dict, Any, Union

from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def load_data(
    file_path: str,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame.
    
    Parameters
    ----------
    file_path : str
        Path to the data file
    **kwargs : dict
        Additional arguments to pass to the pandas read function
        
    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame
        
    Notes
    -----
    Supported file formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - Parquet (.parquet)
    - JSON (.json, .js)
    - Pickle (.pkl, .pickle)
    - Feather (.feather)
    """
    logger.info(f"Loading data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file extension
    _, ext = os.path.splitext(file_path.lower())
    
    try:
        # Load based on file extension
        if ext == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, **kwargs)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path, **kwargs)
        elif ext in ['.json', '.js']:
            df = pd.read_json(file_path, **kwargs)
        elif ext in ['.pkl', '.pickle']:
            df = pd.read_pickle(file_path, **kwargs)
        elif ext == '.feather':
            df = pd.read_feather(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def load_from_url(
    url: str,
    file_format: str = 'csv',
    save_path: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from a URL into a pandas DataFrame.
    
    Parameters
    ----------
    url : str
        URL to the data file
    file_format : str, default 'csv'
        Format of the file ('csv', 'excel', 'json', 'parquet')
    save_path : str, optional
        Path to save the downloaded file. If None, the file is not saved
    **kwargs : dict
        Additional arguments to pass to the pandas read function
        
    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame
    """
    logger.info(f"Loading data from URL: {url}")
    
    try:
        # Download the file
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Download the file
            urllib.request.urlretrieve(url, save_path)
            logger.info(f"Downloaded file to {save_path}")
            
            # Load the file
            return load_data(save_path, **kwargs)
        else:
            # Load directly from URL
            if file_format == 'csv':
                df = pd.read_csv(url, **kwargs)
            elif file_format == 'excel':
                df = pd.read_excel(url, **kwargs)
            elif file_format == 'json':
                df = pd.read_json(url, **kwargs)
            elif file_format == 'parquet':
                # For parquet, we need to download first
                temp_path = 'temp_download.parquet'
                urllib.request.urlretrieve(url, temp_path)
                df = pd.read_parquet(temp_path)
                os.remove(temp_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully loaded data from URL: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
    
    except Exception as e:
        logger.error(f"Error loading data from URL: {str(e)}")
        raise

def read_database(
    query: str,
    connection_string: str,
    engine_type: str = 'sqlalchemy',
    **kwargs
) -> pd.DataFrame:
    """
    Read data from a database into a pandas DataFrame.
    
    Parameters
    ----------
    query : str
        SQL query to execute
    connection_string : str
        Database connection string
    engine_type : str, default 'sqlalchemy'
        Type of database engine to use ('sqlalchemy' or 'sqlite')
    **kwargs : dict
        Additional arguments to pass to the pandas read_sql function
        
    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame
    """
    logger.info("Loading data from database")
    
    try:
        if engine_type == 'sqlalchemy':
            # Import sqlalchemy here to avoid dependencies
            try:
                from sqlalchemy import create_engine
            except ImportError:
                logger.error("Error: SQLAlchemy is required for database connections")
                raise ImportError("SQLAlchemy is required for database connections. "
                                 "Install it with 'pip install sqlalchemy'")
            
            # Create engine and read data
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine, **kwargs)
            
        elif engine_type == 'sqlite':
            # Use built-in sqlite3 module
            import sqlite3
            
            # Connect and read data
            conn = sqlite3.connect(connection_string)
            df = pd.read_sql(query, conn, **kwargs)
            conn.close()
            
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
        
        logger.info(f"Successfully loaded data from database: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from database: {str(e)}")
        raise 