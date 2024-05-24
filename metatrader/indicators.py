import numpy as np

def calculate_ema(prices, period):
    if isinstance(prices, (float, int)):
        # If prices is a single value, return it as is
        return prices
    elif isinstance(prices, (list, np.ndarray, pd.Series)):
        # If prices is a list, array, or pandas Series, calculate the EMA
        ema_values = np.zeros_like(prices)
        ema_values[:period] = np.nan
        sma = np.mean(prices[:period])
        ema_values[period - 1] = sma
        multiplier = 2 / (period + 1)
        for i in range(period, len(prices)):
            ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
        return pd.Series(ema_values, index=prices.index)
    else:
        raise ValueError("Invalid input type for prices. Expected float, int, list, numpy array, or pandas Series.")