"""
Logging utilities for EDA Automator

This module provides logging setup and management for EDA Automator.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, Union

# Default logger
LOGGER_NAME = "eda_automator"
DEFAULT_LOG_LEVEL = logging.INFO

def setup_logging(
    level: Union[int, str] = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for EDA Automator.
    
    Parameters
    ----------
    level : int or str, default logging.INFO
        Logging level
    log_file : str, optional
        Path to log file. If None, no file logging is set up
    console : bool, default True
        Whether to log to console
    format_string : str, optional
        Custom log format string. If None, a default format is used
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(LOGGER_NAME)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set log level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Default format string
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger() -> logging.Logger:
    """
    Get the EDA Automator logger.
    
    If the logger has not been set up, it will be initialized with default settings.
    
    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(LOGGER_NAME)
    
    # If logger has no handlers, set up with defaults
    if not logger.handlers:
        return setup_logging()
        
    return logger 