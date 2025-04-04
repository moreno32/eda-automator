"""
Dataset generation utilities for EDA Automator

This module provides functions for generating synthetic datasets
for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union, List, Tuple
import uuid
import datetime
from eda_automator.core.utils import get_logger

# Initialize logger
logger = get_logger()

def create_dataset(
    size: int = 1000,
    data_type: str = 'basic',
    num_features: int = 10,
    num_categorical: int = 3,
    seed: Optional[int] = None,
    include_missing: bool = True,
    include_outliers: bool = True,
    missing_rate: float = 0.05,
    outlier_rate: float = 0.02,
    correlation_strength: float = 0.7
) -> pd.DataFrame:
    """
    Create a synthetic dataset for testing and demonstration.
    
    Parameters
    ----------
    size : int, default 1000
        Number of samples in the dataset
    data_type : str, default 'basic'
        Type of dataset to generate ('basic', 'timeseries', 'mixed')
    num_features : int, default 10
        Number of features to generate (excluding target variables)
    num_categorical : int, default 3
        Number of categorical features to include
    seed : int, optional
        Random seed for reproducibility
    include_missing : bool, default True
        Whether to include missing values
    include_outliers : bool, default True
        Whether to include outliers
    missing_rate : float, default 0.05
        Proportion of missing values (0-1)
    outlier_rate : float, default 0.02
        Proportion of outliers (0-1)
    correlation_strength : float, default 0.7
        Strength of correlations between features (0-1)
        
    Returns
    -------
    pandas.DataFrame
        Generated synthetic dataset
    """
    logger.info(f"Generating {data_type} dataset with {size} samples and {num_features} features")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Select the appropriate generator based on data_type
    if data_type == 'timeseries':
        return create_timeseries_dataset(
            size=size,
            num_features=num_features,
            seed=seed,
            include_missing=include_missing,
            include_outliers=include_outliers,
            missing_rate=missing_rate,
            outlier_rate=outlier_rate
        )
    elif data_type == 'mixed':
        return create_mixed_dataset(
            size=size,
            num_features=num_features,
            num_categorical=num_categorical,
            seed=seed,
            include_missing=include_missing,
            include_outliers=include_outliers,
            missing_rate=missing_rate,
            outlier_rate=outlier_rate,
            correlation_strength=correlation_strength
        )
    else:  # 'basic' is default
        return create_basic_dataset(
            size=size,
            num_features=num_features,
            num_categorical=num_categorical,
            seed=seed,
            include_missing=include_missing,
            include_outliers=include_outliers,
            missing_rate=missing_rate,
            outlier_rate=outlier_rate,
            correlation_strength=correlation_strength
        )

def create_basic_dataset(
    size: int = 1000,
    num_features: int = 10,
    num_categorical: int = 3,
    seed: Optional[int] = None,
    include_missing: bool = True,
    include_outliers: bool = True,
    missing_rate: float = 0.05,
    outlier_rate: float = 0.02,
    correlation_strength: float = 0.7
) -> pd.DataFrame:
    """
    Create a basic synthetic dataset with numeric and categorical features.
    
    Parameters
    ----------
    size : int, default 1000
        Number of samples in the dataset
    num_features : int, default 10
        Number of features to generate (excluding target variables)
    num_categorical : int, default 3
        Number of categorical features to include
    seed : int, optional
        Random seed for reproducibility
    include_missing : bool, default True
        Whether to include missing values
    include_outliers : bool, default True
        Whether to include outliers
    missing_rate : float, default 0.05
        Proportion of missing values (0-1)
    outlier_rate : float, default 0.02
        Proportion of outliers (0-1)
    correlation_strength : float, default 0.7
        Strength of correlations between features (0-1)
        
    Returns
    -------
    pandas.DataFrame
        Generated synthetic dataset
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Ensure valid parameters
    num_categorical = min(num_categorical, num_features)
    num_numeric = num_features - num_categorical
    
    # Generate IDs
    customer_ids = [str(uuid.uuid4())[:8] for _ in range(size)]
    
    # Generate numeric features with correlations
    numeric_data = np.random.randn(size, num_numeric)
    
    # Add correlations between some features
    if num_numeric >= 3 and correlation_strength > 0:
        # Feature 1 influences feature 2
        numeric_data[:, 1] = correlation_strength * numeric_data[:, 0] + (1 - correlation_strength) * numeric_data[:, 1]
        
        # Features 0 and 1 influence feature 2
        if num_numeric >= 3:
            numeric_data[:, 2] = (correlation_strength/2) * numeric_data[:, 0] + \
                               (correlation_strength/2) * numeric_data[:, 1] + \
                               (1 - correlation_strength) * numeric_data[:, 2]
    
    # Generate categorical features
    categorical_data = []
    for i in range(num_categorical):
        num_categories = np.random.randint(2, 6)  # 2-5 categories
        categorical_data.append(np.random.choice(num_categories, size=size))
    
    # Generate target variables
    
    # Binary target (churn)
    if num_numeric >= 2:
        # Churn influenced by first two numeric features
        churn_prob = 1 / (1 + np.exp(-(0.8 * numeric_data[:, 0] - 0.7 * numeric_data[:, 1])))
        churn = (np.random.random(size) < churn_prob).astype(int)
    else:
        churn = np.random.choice([0, 1], size=size, p=[0.8, 0.2])
    
    # Continuous target (income)
    if num_numeric >= 2:
        # Income influenced by first two numeric features and first categorical if available
        income_base = 50000 + 15000 * numeric_data[:, 0] - 8000 * numeric_data[:, 1]
        if num_categorical > 0:
            for cat_value in range(max(categorical_data[0]) + 1):
                mask = categorical_data[0] == cat_value
                income_base[mask] += (cat_value + 1) * 5000
        
        # Add some noise
        income = income_base + np.random.normal(0, 5000, size)
        income = np.maximum(income, 20000)  # Set minimum income
    else:
        income = 50000 + 15000 * np.random.randn(size)
    
    # Create DataFrame
    df_dict = {'customer_id': customer_ids}
    
    # Add numeric features
    for i in range(num_numeric):
        df_dict[f'feature_{i+1}'] = numeric_data[:, i]
    
    # Add categorical features
    category_names = ['category', 'segment', 'region', 'status', 'plan']
    for i in range(num_categorical):
        name = category_names[i] if i < len(category_names) else f'category_{i+1}'
        df_dict[name] = [f'Cat{val}' for val in categorical_data[i]]
    
    # Add target variables
    df_dict['churn'] = churn
    df_dict['income'] = income
    
    # Create DataFrame
    df = pd.DataFrame(df_dict)
    
    # Add timestamps (registration date)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365*2)  # 2 years ago
    
    df['registration_date'] = [
        start_date + datetime.timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        ) for _ in range(size)
    ]
    
    # Add missing values if requested
    if include_missing and missing_rate > 0:
        # Don't add missing values to ID or target columns
        exclude_cols = ['customer_id', 'churn', 'income', 'registration_date']
        for col in df.columns:
            if col not in exclude_cols:
                # Randomly set values to NaN
                mask = np.random.random(size) < missing_rate
                df.loc[mask, col] = np.nan
    
    # Add outliers if requested
    if include_outliers and outlier_rate > 0:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != 'churn':  # Don't add outliers to binary target
                # Add outliers to random positions
                mask = np.random.random(size) < outlier_rate
                if col == 'income':
                    # Extreme incomes
                    df.loc[mask, col] = df.loc[mask, col] * np.random.choice([3, 4, 5], size=mask.sum())
                else:
                    # Extreme values (3-5 standard deviations from mean)
                    mean = df[col].mean()
                    std = df[col].std()
                    df.loc[mask, col] = mean + std * np.random.choice([-5, -4, -3, 3, 4, 5], size=mask.sum())
    
    logger.info(f"Basic dataset created with shape {df.shape}")
    return df

def create_timeseries_dataset(
    size: int = 1000,
    num_features: int = 5,
    periods: int = 100,
    freq: str = 'D',
    seed: Optional[int] = None,
    include_missing: bool = True,
    include_outliers: bool = True,
    missing_rate: float = 0.05,
    outlier_rate: float = 0.02
) -> pd.DataFrame:
    """
    Create a synthetic time series dataset.
    
    Parameters
    ----------
    size : int, default 1000
        Number of unique entities (e.g., customers, products)
    num_features : int, default 5
        Number of features to generate (excluding target variables)
    periods : int, default 100
        Number of time periods
    freq : str, default 'D'
        Frequency of time periods ('D' for days, 'H' for hours, etc.)
    seed : int, optional
        Random seed for reproducibility
    include_missing : bool, default True
        Whether to include missing values
    include_outliers : bool, default True
        Whether to include outliers
    missing_rate : float, default 0.05
        Proportion of missing values (0-1)
    outlier_rate : float, default 0.02
        Proportion of outliers (0-1)
        
    Returns
    -------
    pandas.DataFrame
        Generated synthetic time series dataset
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate time index
    end_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(end=end_date, periods=periods, freq=freq)
    
    # Generate entity IDs
    entity_ids = [f'entity_{i+1:03d}' for i in range(size)]
    
    # Generate trends, seasonality, and noise components
    trends = {}
    seasonalities = {}
    noises = {}
    
    # Limit to a reasonable number to avoid memory issues
    actual_size = min(size, 100)
    actual_periods = min(periods, 365)
    
    if actual_size < size or actual_periods < periods:
        logger.warning(f"Limiting timeseries to {actual_size} entities and {actual_periods} periods to avoid memory issues")
    
    entity_ids = entity_ids[:actual_size]
    dates = dates[-actual_periods:]
    
    # Generate base data
    data_frames = []
    
    for entity_id in entity_ids:
        # Generate features
        entity_data = {'entity_id': entity_id, 'date': dates}
        
        # Linear trend with random slope
        trend = np.linspace(0, np.random.uniform(5, 15), len(dates))
        
        # Seasonality (sine wave with random phase and amplitude)
        amplitude = np.random.uniform(1, 5)
        phase = np.random.uniform(0, 2*np.pi)
        period = len(dates) / np.random.randint(3, 7)  # 3-6 cycles
        seasonality = amplitude * np.sin(2*np.pi*np.arange(len(dates))/period + phase)
        
        # Generate target variable (e.g., sales, revenue)
        noise = np.random.normal(0, 0.5, len(dates))
        target = 100 + trend + seasonality + noise
        entity_data['income_ts'] = target
        
        # Generate additional features
        for i in range(num_features):
            # Feature with some correlation to target
            correlation = np.random.uniform(0.3, 0.8) * (1 if np.random.random() > 0.3 else -1)
            feature = correlation * target + (1 - abs(correlation)) * np.random.normal(50, 10, len(dates))
            
            # Add feature-specific trend and seasonality
            if i % 3 == 0:  # One-third of features have different trend
                feature_trend = np.linspace(0, np.random.uniform(-10, 10), len(dates))
                feature += feature_trend
            
            if i % 2 == 0:  # Half of features have different seasonality
                feature_seasonality = np.random.uniform(0.5, 2) * np.sin(
                    2*np.pi*np.arange(len(dates))/(len(dates)/np.random.randint(2, 5)) + np.random.uniform(0, 2*np.pi)
                )
                feature += feature_seasonality
            
            entity_data[f'feature_ts_{i+1}'] = feature
        
        # Convert to DataFrame and add to list
        entity_df = pd.DataFrame(entity_data)
        data_frames.append(entity_df)
    
    # Combine all entities into one DataFrame
    df = pd.concat(data_frames, ignore_index=True)
    
    # Add some categorical features
    df['segment'] = np.random.choice(['Segment A', 'Segment B', 'Segment C'], size=len(df))
    df['region'] = np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=len(df))
    
    # Add missing values if requested
    if include_missing and missing_rate > 0:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            mask = np.random.random(len(df)) < missing_rate
            df.loc[mask, col] = np.nan
    
    # Add outliers if requested
    if include_outliers and outlier_rate > 0:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            mask = np.random.random(len(df)) < outlier_rate
            mean = df[col].mean()
            std = df[col].std()
            df.loc[mask, col] = mean + std * np.random.choice([-5, -4, -3, 3, 4, 5], size=mask.sum())
    
    logger.info(f"Time series dataset created with shape {df.shape}")
    return df

def create_mixed_dataset(
    size: int = 1000,
    num_features: int = 10,
    num_categorical: int = 3,
    seed: Optional[int] = None,
    include_missing: bool = True,
    include_outliers: bool = True,
    missing_rate: float = 0.05,
    outlier_rate: float = 0.02,
    correlation_strength: float = 0.7
) -> pd.DataFrame:
    """
    Create a dataset with mixed data types including text and dates.
    
    Parameters
    ----------
    size : int, default 1000
        Number of samples in the dataset
    num_features : int, default 10
        Number of features to generate (excluding target variables)
    num_categorical : int, default 3
        Number of categorical features to include
    seed : int, optional
        Random seed for reproducibility
    include_missing : bool, default True
        Whether to include missing values
    include_outliers : bool, default True
        Whether to include outliers
    missing_rate : float, default 0.05
        Proportion of missing values (0-1)
    outlier_rate : float, default 0.02
        Proportion of outliers (0-1)
    correlation_strength : float, default 0.7
        Strength of correlations between features (0-1)
        
    Returns
    -------
    pandas.DataFrame
        Generated synthetic dataset with mixed data types
    """
    # Create a basic dataset first
    df = create_basic_dataset(
        size=size,
        num_features=max(num_features - 4, 3),  # Reserve space for additional features
        num_categorical=num_categorical,
        seed=seed,
        include_missing=include_missing,
        include_outliers=include_outliers,
        missing_rate=missing_rate,
        outlier_rate=outlier_rate,
        correlation_strength=correlation_strength
    )
    
    # Add email addresses
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'example.com']
    df['email'] = [
        f"{first_name.lower()}.{last_name.lower()}@{np.random.choice(domains)}"
        for first_name, last_name in zip(
            np.random.choice(['John', 'Jane', 'Robert', 'Mary', 'David', 'Linda', 'Michael', 'Sarah', 'William', 'Susan'], size),
            np.random.choice(['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor'], size)
        )
    ]
    
    # Add IP addresses
    df['ip_address'] = [
        f"{np.random.randint(1, 256)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}"
        for _ in range(size)
    ]
    
    # Add URLs
    websites = ['example.com', 'test.org', 'demo.net', 'sample.io', 'trial.tech']
    df['website'] = [
        f"https://www.{np.random.choice(websites)}/{uuid.uuid4().hex[:8]}"
        for _ in range(size)
    ]
    
    # Add additional date - last activity
    df['last_activity'] = [
        df.loc[i, 'registration_date'] + datetime.timedelta(days=np.random.randint(1, 365))
        for i in range(size)
    ]
    
    logger.info(f"Mixed dataset created with shape {df.shape}")
    return df