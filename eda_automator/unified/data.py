"""
Data loading and validation functions for unified EDA reports.

This module contains functions for loading and validating data
from various file formats.
"""

import os
import pandas as pd
import numpy as np
from .config import SUPPORTED_FILE_EXTENSIONS

def load_data(file_path):
    """
    Load data from various file formats.
    
    Args:
        file_path (str): Path to the data file
    
    Returns:
        pandas.DataFrame: Loaded data
    
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # CSV files
    if file_ext in SUPPORTED_FILE_EXTENSIONS["csv"]:
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded CSV file with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            # Try with different encodings
            try:
                df = pd.read_csv(file_path, encoding='latin1')
                print(f"Loaded CSV file (latin1 encoding) with {df.shape[0]} rows and {df.shape[1]} columns")
                return df
            except Exception as e2:
                raise ValueError(f"Error loading CSV file: {str(e)} and {str(e2)}")
    
    # Excel files
    elif file_ext in SUPPORTED_FILE_EXTENSIONS["excel"]:
        try:
            df = pd.read_excel(file_path)
            print(f"Loaded Excel file with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading Excel file: {str(e)}")
    
    # Parquet files
    elif file_ext in SUPPORTED_FILE_EXTENSIONS["parquet"]:
        try:
            df = pd.read_parquet(file_path)
            print(f"Loaded Parquet file with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading Parquet file: {str(e)}")
    
    # JSON files
    elif file_ext in SUPPORTED_FILE_EXTENSIONS["json"]:
        try:
            df = pd.read_json(file_path)
            print(f"Loaded JSON file with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {str(e)}")
    
    # Pickle files
    elif file_ext in SUPPORTED_FILE_EXTENSIONS["pickle"]:
        try:
            df = pd.read_pickle(file_path)
            print(f"Loaded Pickle file with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading Pickle file: {str(e)}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def validate_data(df):
    """
    Validate the data before analysis.
    
    Args:
        df (pandas.DataFrame): Data to validate
    
    Raises:
        ValueError: If data does not meet minimum requirements
    """
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot proceed with analysis.")
    
    # Check if there are any columns
    if df.shape[1] == 0:
        raise ValueError("DataFrame has no columns. Cannot proceed with analysis.")
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        print(f"Warning: DataFrame contains duplicate column names: {duplicate_cols}")
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        print(f"Warning: DataFrame contains completely empty columns: {empty_cols}")
    
    # Check for column names with spaces or special characters
    problematic_cols = [col for col in df.columns if ' ' in col or any(c in col for c in '!@#$%^&*()+={}[]|\\:;\"\'<>,.?/')]
    if problematic_cols:
        print(f"Warning: DataFrame contains columns with spaces or special characters: {problematic_cols}")
    
    # Convert date-like object columns to datetime
    for col in df.select_dtypes(include=['object']):
        try:
            # Check if column looks like dates
            if df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').all():
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"Converted column '{col}' to datetime")
        except Exception:
            pass
    
    # Set DataFrame name to filename if possible
    if getattr(df, 'name', None) is None:
        df.name = "Dataset"
        
    return df

def get_column_types(df):
    """
    Classify columns by data type.
    
    Args:
        df (pandas.DataFrame): Data to classify
    
    Returns:
        dict: Dictionary with column types
    """
    column_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'boolean': [],
        'id': []
    }
    
    for col in df.columns:
        # Get column data type
        dtype = df[col].dtype
        
        # Check if column is numeric
        if np.issubdtype(dtype, np.number):
            n_unique = df[col].nunique()
            
            # If numeric column has few unique values, it might be categorical
            if n_unique <= 10 and n_unique / len(df) < 0.05:
                column_types['categorical'].append(col)
            # Check if it might be an ID column
            elif n_unique == len(df) or n_unique / len(df) > 0.9:
                column_types['id'].append(col)
            else:
                column_types['numeric'].append(col)
        
        # Check if column is datetime
        elif np.issubdtype(dtype, np.datetime64):
            column_types['datetime'].append(col)
        
        # Check if column is boolean
        elif dtype == bool:
            column_types['boolean'].append(col)
        
        # Check if column is object or category
        elif dtype == 'object' or dtype.name == 'category':
            n_unique = df[col].nunique()
            
            # If column has many unique values, it might be text
            if n_unique / len(df) > 0.5 and n_unique > 100:
                column_types['text'].append(col)
            # If column has few unique values, it's categorical
            else:
                column_types['categorical'].append(col)
    
    return column_types

def create_dataset(size=1000, data_type='basic', seed=42):
    """
    Create a synthetic dataset for testing EDA functions.
    
    Args:
        size (int): Number of rows in the dataset
        data_type (str): Type of dataset to create ('basic' or 'timeseries')
        seed (int): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Synthetic dataset
    """
    np.random.seed(seed)
    
    if data_type == 'timeseries':
        return _create_timeseries_dataset(size)
    else:
        return _create_basic_dataset(size)

def _create_basic_dataset(size):
    """
    Create a basic synthetic dataset with various data types.
    
    Args:
        size (int): Number of rows in the dataset
        
    Returns:
        pandas.DataFrame: Synthetic dataset
    """
    print(f"Creating synthetic dataset with {size} rows...")
    
    # Create customer IDs
    customer_ids = np.arange(1, size + 1)
    
    # Create categorical variables
    genders = np.random.choice(['Male', 'Female', 'Non-binary'], size=size, p=[0.48, 0.48, 0.04])
    
    education_levels = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD', 'Other'],
        size=size,
        p=[0.3, 0.4, 0.2, 0.05, 0.05]
    )
    
    # Create location data with some correlation between city and country
    countries = np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia'], size=size)
    
    # Dictionary mapping countries to their cities
    country_cities = {
        'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Ottawa'],
        'UK': ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Liverpool'],
        'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne'],
        'France': ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Nice'],
        'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide']
    }
    
    # Generate cities based on countries
    cities = []
    for country in countries:
        cities.append(np.random.choice(country_cities[country]))
    
    # Create numerical variables with some correlations
    age = np.random.normal(35, 12, size=size).astype(int)
    # Limit age to reasonable values
    age = np.clip(age, 18, 90)
    
    # Income correlated with age and education
    base_income = np.random.lognormal(10.5, 0.5, size=size)
    age_factor = (age - 18) / 50  # Normalized age factor
    education_factor = np.zeros(size)
    
    for i, edu in enumerate(education_levels):
        if edu == 'High School':
            education_factor[i] = 0.7
        elif edu == 'Bachelor':
            education_factor[i] = 1.0
        elif edu == 'Master':
            education_factor[i] = 1.3
        elif edu == 'PhD':
            education_factor[i] = 1.6
        else:
            education_factor[i] = 0.8
    
    income = base_income * (0.5 + 0.5 * age_factor) * education_factor
    
    # Create some missing values
    missing_mask = np.random.random(size=size) < 0.05
    income[missing_mask] = np.nan
    
    # Satisfaction score (1-10)
    satisfaction = np.random.normal(7.5, 1.5, size=size)
    satisfaction = np.clip(satisfaction, 1, 10).round(1)
    
    # Years as customer
    tenure = np.random.gamma(5, 1, size=size).round(1)
    tenure = np.clip(tenure, 0.1, 30)
    
    # Products owned (1-5)
    products = np.random.poisson(2, size=size) + 1
    products = np.clip(products, 1, 5)
    
    # Monthly spend
    monthly_spend = income * np.random.beta(2, 5, size=size) / 100
    
    # Last purchase (days ago)
    last_purchase = np.random.exponential(30, size=size).astype(int)
    
    # Website visits per month
    website_visits = np.random.poisson(5, size=size)
    
    # Support tickets
    support_tickets = np.random.poisson(0.5, size=size)
    
    # Target variable (churn probability)
    # Factors that increase churn: low satisfaction, low tenure, high support tickets
    churn_score = (
        (10 - satisfaction) / 10 * 0.5 +
        np.exp(-tenure / 3) * 0.3 +
        np.tanh(support_tickets) * 0.2
    )
    
    # Add some noise
    churn_score = churn_score + np.random.normal(0, 0.1, size=size)
    churn_score = np.clip(churn_score, 0, 1)
    
    # Binary churn (using threshold)
    churn = (churn_score > 0.5).astype(int)
    
    # Create email addresses (for text analysis)
    emails = []
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
    
    for i in range(size):
        name_part = f"customer{customer_ids[i]}"
        domain = np.random.choice(domains)
        emails.append(f"{name_part}@{domain}")
    
    # Create dates
    today = pd.Timestamp.today()
    registration_dates = [
        today - pd.Timedelta(days=int(t * 365)) 
        for t in tenure
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'gender': genders,
        'education': education_levels,
        'country': countries,
        'city': cities,
        'income': income,
        'satisfaction': satisfaction,
        'tenure_years': tenure,
        'products': products,
        'monthly_spend': monthly_spend,
        'last_purchase_days': last_purchase,
        'website_visits': website_visits,
        'support_tickets': support_tickets,
        'churn_score': churn_score,
        'churn': churn,
        'email': emails,
        'registration_date': registration_dates
    })
    
    # Add some outliers
    if size > 10:
        outlier_indices = np.random.choice(size, size=int(size * 0.02), replace=False)
        df.loc[outlier_indices, 'income'] = df['income'].max() * np.random.uniform(2, 5, size=len(outlier_indices))
        df.loc[outlier_indices, 'monthly_spend'] = df['monthly_spend'].max() * np.random.uniform(2, 5, size=len(outlier_indices))
    
    # Set name attribute
    df.name = "Customer Dataset"
    
    print(f"Synthetic dataset created with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def _create_timeseries_dataset(size):
    """
    Create a synthetic time series dataset.
    
    Args:
        size (int): Number of rows in the dataset
        
    Returns:
        pandas.DataFrame: Synthetic time series dataset
    """
    print(f"Creating synthetic time series dataset with {size} rows...")
    
    # Number of unique entities (e.g., customers, products)
    n_entities = min(100, size // 10)
    
    # Entity IDs
    entity_ids = np.arange(1, n_entities + 1)
    
    # Time periods
    max_periods = size // n_entities + 1
    
    # Generate data
    data = []
    
    # Current date as the end date
    end_date = pd.Timestamp.today()
    
    # Start date (1 year ago)
    start_date = end_date - pd.Timedelta(days=365)
    
    # Generate dates between start and end
    date_range = pd.date_range(start=start_date, end=end_date, periods=max_periods)
    
    for entity_id in entity_ids:
        # Base values for this entity
        base_income = np.random.lognormal(8, 0.5)
        base_cost = base_income * np.random.uniform(0.3, 0.7)
        trend = np.random.choice([-0.1, 0, 0.1, 0.2])
        seasonality = np.random.uniform(0.05, 0.2)
        
        for i, date in enumerate(date_range):
            # Time component for trend
            time_factor = 1 + trend * (i / max_periods)
            
            # Seasonal component (yearly cycle)
            month = date.month
            season = np.sin(2 * np.pi * month / 12)
            seasonal_factor = 1 + seasonality * season
            
            # Add noise
            noise_income = np.random.normal(0, 0.05)
            noise_cost = np.random.normal(0, 0.03)
            
            # Calculate values
            income = base_income * time_factor * seasonal_factor * (1 + noise_income)
            cost = base_cost * time_factor * (1 + noise_cost)
            profit = income - cost
            
            # Add row to data
            data.append({
                'entity_id': entity_id,
                'date': date,
                'income_ts': income,
                'cost': cost,
                'profit': profit,
                'month': month,
                'day_of_week': date.dayofweek,
                'is_weekend': date.dayofweek >= 5
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Randomly drop some rows to simulate missing data
    if len(df) > 10:
        drop_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
        df = df.drop(drop_indices)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Limit to requested size
    df = df.head(size)
    
    # Set name attribute
    df.name = "Time Series Dataset"
    
    print(f"Synthetic time series dataset created with {df.shape[0]} rows and {df.shape[1]} columns")
    return df 