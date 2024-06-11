import pandas as pd
import numpy as np
import unittest

def calculate_ema(prices, period):
    # Ensure that the series is numeric
    prices = pd.to_numeric(prices, errors='coerce')

    if isinstance(prices, (float, int)):
        return prices
    elif isinstance(prices, (list, np.ndarray, pd.Series)):
        ema_values = np.full(len(prices), np.nan, dtype=np.float64)
        if len(prices) < period:
            return pd.Series(ema_values, index=prices.index)
        
        sma = np.mean(prices[:period])
        ema_values[period - 1] = sma
        multiplier = 2 / (period + 1)
        for i in range(period, len(prices)):
            ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
        ema_series = pd.Series(ema_values, index=prices.index)
        print(f"EMA: {ema_series}")
        return ema_series
    else:
        raise ValueError("Invalid input type for prices. Expected float, int, list, numpy array, or pandas Series.")

class TestStrategy(unittest.TestCase):

    def test_calculate_ema_non_numeric(self):
        prices = pd.Series(['abc', 'def', 'ghi'])
        period = 3
        result = calculate_ema(prices, period)
        expected_ema = pd.Series([np.nan, np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected_ema, check_names=False)

if __name__ == '__main__':
    unittest.main()
