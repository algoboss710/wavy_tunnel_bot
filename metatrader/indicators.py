import pandas as pd
import numpy as np
import logging

def calculate_ema(prices, period):
    if not isinstance(prices, (list, np.ndarray, pd.Series)):
        raise ValueError("Invalid input type for prices. Expected list, numpy array, or pandas Series.")
    
    logging.debug(f"Calculating EMA for period: {period}, prices: {prices}")
    
    # Convert input to a pandas Series to ensure consistency
    prices = pd.Series(prices)
    
    # Ensure that the series is numeric
    prices = pd.to_numeric(prices, errors='coerce')
    logging.debug(f"Prices converted to numeric: {prices}")

    ema_values = np.full(len(prices), np.nan, dtype=np.float64)
    if len(prices) < period:
        return pd.Series(ema_values, index=prices.index)
    
    sma = np.mean(prices[:period])
    ema_values[period - 1] = sma
    logging.debug(f"Initial SMA: {sma}")
    
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
        logging.debug(f"EMA value at index {i}: {ema_values[i]}")
    
    ema_series = pd.Series(ema_values, index=prices.index)
    return ema_series
