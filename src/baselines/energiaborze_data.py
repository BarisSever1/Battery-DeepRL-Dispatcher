"""Data loader for Energiabőrze policy from Excel file."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# Cache for loaded data
_ENERGIABORZE_DATA_CACHE: Dict[str, Dict[int, Dict[str, float]]] = {}


def load_energiaborze_data(excel_path: str | Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Load Energiabőrze discharge/charge data from Excel file.
    
    Converts 15-minute data to hourly aggregates by summing discharge and charge
    values for each hour.
    
    Args:
        excel_path: Path to Excel file with columns: date, discharge, charge
        
    Returns:
        Dictionary mapping date strings (YYYY-MM-DD) to hourly data:
        {date_str: {hour: {'discharge': MW, 'charge': MW}}}
    """
    global _ENERGIABORZE_DATA_CACHE
    
    excel_path = Path(excel_path)
    cache_key = str(excel_path.absolute())
    
    # Return cached data if available
    if cache_key in _ENERGIABORZE_DATA_CACHE:
        return _ENERGIABORZE_DATA_CACHE[cache_key]
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Energiabőrze Excel file not found: {excel_path}")
    
    # Load Excel file
    df = pd.read_excel(excel_path)
    
    # Expected columns: date, discharge, charge
    # Handle different possible column names
    date_col = None
    discharge_col = None
    charge_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'date' in col_lower or 'datum' in col_lower:
            date_col = col
        elif 'discharge' in col_lower or 'disch' in col_lower:
            discharge_col = col
        elif 'charge' in col_lower or 'chrg' in col_lower:
            charge_col = col
    
    if date_col is None or discharge_col is None or charge_col is None:
        raise ValueError(
            f"Could not find required columns in Excel file. "
            f"Found columns: {df.columns.tolist()}. "
            f"Need: date, discharge, charge"
        )
    
    # Parse date column
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract date and hour from datetime
    df['date_only'] = df[date_col].dt.date
    df['hour'] = df[date_col].dt.hour
    
    # Group by date and hour, sum discharge and charge values
    hourly_data = df.groupby(['date_only', 'hour']).agg({
        discharge_col: 'sum',
        charge_col: 'sum'
    }).reset_index()
    
    # Convert to dictionary structure
    result: Dict[str, Dict[int, Dict[str, float]]] = {}
    
    for _, row in hourly_data.iterrows():
        date_str = row['date_only'].strftime('%Y-%m-%d')
        hour = int(row['hour'])
        discharge_mw = float(row[discharge_col])
        charge_mw = float(row[charge_col])
        
        if date_str not in result:
            result[date_str] = {}
        
        result[date_str][hour] = {
            'discharge': discharge_mw,
            'charge': charge_mw
        }
    
    # Cache the result
    _ENERGIABORZE_DATA_CACHE[cache_key] = result
    
    return result


def get_energiaborze_hourly_data(
    excel_path: str | Path,
    date_str: str
) -> Dict[int, Dict[str, float]]:
    """
    Get hourly discharge/charge data for a specific date.
    
    Args:
        excel_path: Path to Excel file
        date_str: Date string in format 'YYYY-MM-DD'
        
    Returns:
        Dictionary mapping hour (0-23) to {'discharge': MW, 'charge': MW}
        Returns empty dict if date not found
    """
    data = load_energiaborze_data(excel_path)
    return data.get(date_str, {})

