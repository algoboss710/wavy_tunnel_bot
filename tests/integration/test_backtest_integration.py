import unittest
import pandas as pd
from backtesting.backtest import run_backtest
from strategy.tunnel_strategy import check_entry_conditions

class TestBacktestIntegration(unittest.TestCase):

    def test_run_backtest_with_dummy_data(self):
        # Prepare dummy data
        dummy_data = pd.DataFrame({
            'time': pd.date_range(start='2024-06-17', periods=5, freq='h'),
            'open': [1.1, 1.2, 1.3, 1.4, 1.5],
            'high': [1.15, 1.25, 1.35, 1.45, 1.55],
            'low': [1.05, 1.15, 1.25, 1.35, 1.45],
            'close': [1.1, 1.2, 1.3, 1.4, 1.5],
            'tick_volume': [100, 200, 300, 400, 500],
            'spread': [1, 1, 1, 1, 1],
            'real_volume': [10, 20, 30, 40, 50]
        })

        # Call run_backtest with dummy data
        result = run_backtest('EURUSD', dummy_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)
        print(f"Result: {result}")

        # Assert the results
        self.assertIsNotNone(result, "run_backtest returned None")
        self.assertTrue(result['buy_condition'], f"Expected buy_condition to be True, but got {result['buy_condition']}")
        self.assertFalse(result['sell_condition'], f"Expected sell_condition to be False, but got {result['sell_condition']}")

    def test_run_backtest_edge_cases(self):
        # Prepare edge case data
        extreme_data = pd.DataFrame({
            'time': pd.date_range(start='2024-06-17', periods=10000, freq='h'),
            'open': [1.1] * 10000,
            'high': [1.15] * 10000,
            'low': [1.05] * 10000,
            'close': [1.1] * 10000,
            'tick_volume': [100] * 10000,
            'spread': [1] * 10000,
            'real_volume': [10] * 10000
        })

        # Call run_backtest with edge case data
        result = run_backtest('EURUSD', extreme_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)
        print(f"Result: {result}")

        # Assert the results
        self.assertIsNotNone(result, "run_backtest returned None")

    def test_check_entry_conditions_direct(self):
        # Prepare data for check_entry_conditions
        data = pd.DataFrame({
            'time': pd.date_range(start='2024-06-17', periods=5, freq='h'),
            'open': [1.1, 1.2, 1.3, 1.4, 1.5],
            'high': [1.15, 1.25, 1.35, 1.45, 1.55],
            'low': [1.05, 1.15, 1.25, 1.35, 1.45],
            'close': [1.1, 1.2, 1.3, 1.4, 1.5],
            'tick_volume': [100, 200, 300, 400, 500],
            'spread': [1, 1, 1, 1, 1],
            'real_volume': [10, 20, 30, 40, 50]
        })

        peaks, dips = [], []

        for i in range(5):
            row = data.iloc[i]
            print(f"Calling check_entry_conditions for row {i}")
            result = check_entry_conditions(row, peaks, dips, 'EURUSD')
            print(f"Result for row {i}: {result}, type: {type(result)}")
            print(f"Result[0]: {result[0]}, Result[1]: {result[1]}")
            self.assertFalse(result[0])
            self.assertFalse(result[1])

    def test_performance_and_stability(self):
        # Prepare large dataset
        large_data = pd.DataFrame({
            'time': pd.date_range(start='2024-06-17', periods=100000, freq='h'),
            'open': [1.1] * 100000,
            'high': [1.15] * 100000,
            'low': [1.05] * 100000,
            'close': [1.1] * 100000,
            'tick_volume': [100] * 100000,
            'spread': [1] * 100000,
            'real_volume': [10] * 100000
        })

        # Call run_backtest with large dataset
        result = run_backtest('EURUSD', large_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)
        print(f"Result: {result}")

        # Assert the results
        self.assertIsNotNone(result, "run_backtest returned None")

if __name__ == '__main__':
    unittest.main()
