"""
Formatting utilities for EDA Automator

This module provides formatting functions for numerical values, percentages,
dates, and other types of data used in EDA Automator reports.
"""

import locale
from typing import Union, Optional, Any, List, Dict

def format_number(
    value: Union[int, float],
    decimal_places: int = 2,
    use_thousands_separator: bool = True,
    prefix: str = "",
    suffix: str = "",
    nan_string: str = "N/A"
) -> str:
    """
    Format a number with the specified decimal places and thousands separator.
    
    Parameters
    ----------
    value : int or float
        Number to format
    decimal_places : int, default 2
        Number of decimal places to display
    use_thousands_separator : bool, default True
        Whether to use thousands separator
    prefix : str, default ""
        Prefix to add before the number
    suffix : str, default ""
        Suffix to add after the number
    nan_string : str, default "N/A"
        String to use for NaN values
        
    Returns
    -------
    str
        Formatted number string
    """
    # Check for NaN values
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return nan_string
    
    # Format with locale-specific rules
    try:
        if isinstance(value, int):
            if use_thousands_separator:
                formatted = locale.format_string("%d", value, grouping=True)
            else:
                formatted = str(value)
        else:
            format_string = f"%.{decimal_places}f"
            if use_thousands_separator:
                formatted = locale.format_string(format_string, value, grouping=True)
            else:
                formatted = format_string % value
                
        return f"{prefix}{formatted}{suffix}"
    except (ValueError, TypeError):
        # Fallback to basic formatting
        if isinstance(value, int):
            return f"{prefix}{value}{suffix}"
        else:
            return f"{prefix}{value:.{decimal_places}f}{suffix}"

def format_percent(
    value: float,
    decimal_places: int = 1,
    include_symbol: bool = True,
    nan_string: str = "N/A"
) -> str:
    """
    Format a value as a percentage.
    
    Parameters
    ----------
    value : float
        Value to format as percentage (0.1 = 10%)
    decimal_places : int, default 1
        Number of decimal places to display
    include_symbol : bool, default True
        Whether to include the % symbol
    nan_string : str, default "N/A"
        String to use for NaN values
        
    Returns
    -------
    str
        Formatted percentage string
    """
    # Check for NaN values
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return nan_string
    
    # Convert to percentage and format
    percentage = value * 100
    
    # Format with locale-specific rules
    try:
        format_string = f"%.{decimal_places}f"
        if include_symbol:
            return f"{locale.format_string(format_string, percentage, grouping=True)}%"
        else:
            return locale.format_string(format_string, percentage, grouping=True)
    except (ValueError, TypeError):
        # Fallback to basic formatting
        if include_symbol:
            return f"{percentage:.{decimal_places}f}%"
        else:
            return f"{percentage:.{decimal_places}f}"

def format_list(
    items: List[Any],
    separator: str = ", ",
    max_items: Optional[int] = None,
    more_text: str = "... and {n} more"
) -> str:
    """
    Format a list of items as a string.
    
    Parameters
    ----------
    items : list
        List of items to format
    separator : str, default ", "
        Separator to use between items
    max_items : int, optional
        Maximum number of items to include. If None, all items are included
    more_text : str, default "... and {n} more"
        Template for text to append when there are more items than max_items.
        {n} will be replaced with the number of omitted items
        
    Returns
    -------
    str
        Formatted list string
    """
    if not items:
        return ""
    
    if max_items is not None and len(items) > max_items:
        visible_items = items[:max_items]
        hidden_count = len(items) - max_items
        return separator.join(str(item) for item in visible_items) + " " + more_text.format(n=hidden_count)
    else:
        return separator.join(str(item) for item in items)

def format_dict(
    data: Dict[str, Any],
    key_value_separator: str = ": ",
    item_separator: str = ", ",
    max_items: Optional[int] = None,
    more_text: str = "... and {n} more"
) -> str:
    """
    Format a dictionary as a string.
    
    Parameters
    ----------
    data : dict
        Dictionary to format
    key_value_separator : str, default ": "
        Separator to use between keys and values
    item_separator : str, default ", "
        Separator to use between items
    max_items : int, optional
        Maximum number of items to include. If None, all items are included
    more_text : str, default "... and {n} more"
        Template for text to append when there are more items than max_items.
        {n} will be replaced with the number of omitted items
        
    Returns
    -------
    str
        Formatted dictionary string
    """
    if not data:
        return ""
    
    formatted_items = [f"{key}{key_value_separator}{value}" for key, value in data.items()]
    
    return format_list(
        formatted_items,
        separator=item_separator,
        max_items=max_items,
        more_text=more_text
    ) 