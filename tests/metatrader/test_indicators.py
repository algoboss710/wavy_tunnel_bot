import unittest
import numpy as np
import pandas as pd
from metatrader.indicators import calculate_ema

class TestIndicators(unittest.TestCase):
    def test_calculate_ema_list(self):
        data = [100, 200, 300, 400, 500]
        period = 3
        expected_ema = [np.nan, np.nan, 200.0, 300.0, 400.0]
        result = calculate_ema(data, period).tolist()
        np.testing.assert_array_almost_equal(result, expected_ema, decimal=5)

    def test_calculate_ema_numpy_array(self):
        data = np.array([100, 200, 300, 400, 500])
        period = 3
        expected_ema = [np.nan, np.nan, 200.0, 300.0, 400.0]
        result = calculate_ema(data, period).tolist()
        np.testing.assert_array_almost_equal(result, expected_ema, decimal=5)

    def test_calculate_ema_pandas_series(self):
        data = pd.Series([100, 200, 300, 400, 500])
        period = 3
        expected_ema = pd.Series([np.nan, np.nan, 200.0, 300.0, 400.0])
        result = calculate_ema(data, period)
        pd.testing.assert_series_equal(result, expected_ema)

    def test_calculate_ema_period_greater_than_length(self):
        data = [100, 200, 300]
        period = 5
        expected_ema = [np.nan, np.nan, np.nan]
        result = calculate_ema(data, period).tolist()
        np.testing.assert_array_almost_equal(result, expected_ema, decimal=5)

    def test_calculate_ema_empty_list(self):
        data = []
        period = 3
        expected_ema = []
        result = calculate_ema(data, period).tolist()
        self.assertEqual(result, expected_ema)

    def test_calculate_ema_invalid_input_type(self):
        data = "invalid input"
        period = 3
        with self.assertRaises(ValueError):
            calculate_ema(data, period)

if __name__ == '__main__':
    unittest.main()
