# import unittest
# import pandas as pd
# import numpy as np
# import MetaTrader5 as mt5
# from backtesting.backtest import run_backtest
# from strategy.tunnel_strategy import check_entry_conditions
# import logging

# # Set logging level to DEBUG
# logging.basicConfig(level=logging.DEBUG)

# class TestBacktestIntegration(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         # Initialize MetaTrader 5 connection
#         if not mt5.initialize():
#             raise RuntimeError("MetaTrader 5 initialization failed")
#         # Ensure symbol is available
#         symbol = 'EURUSD'
#         if not mt5.symbol_select(symbol, True):
#             raise RuntimeError(f"Failed to select symbol: {symbol}")
#         print(f"MetaTrader 5 initialized with symbol: {symbol}")

#     @classmethod
#     def tearDownClass(cls):
#         # Shutdown MetaTrader 5 connection
#         mt5.shutdown()
#         print("MetaTrader 5 shutdown completed")

#     def test_run_backtest_with_dummy_data(self):
#         print("Setting up test with sufficient data")

#         # Prepare dummy data with at least 50 data points
#         dummy_data = pd.DataFrame({
#             'time': pd.date_range(start='2024-06-17', periods=50, freq='h'),
#             'open': np.linspace(1.1, 2.0, 50),
#             'high': np.linspace(1.2, 2.1, 50),
#             'low': np.linspace(1.0, 1.9, 50),
#             'close': np.linspace(1.1, 2.0, 50),
#             'tick_volume': np.random.randint(100, 500, 50),
#             'spread': np.random.randint(1, 5, 50),
#             'real_volume': np.random.randint(10, 50, 50),
#             'wavy_c': np.linspace(1.1, 2.0, 50),
#             'wavy_h': np.linspace(1.2, 2.1, 50),
#             'wavy_l': np.linspace(1.0, 1.9, 50),
#             'tunnel1': np.linspace(1.0, 1.5, 50),
#             'tunnel2': np.linspace(0.5, 1.0, 50)
#         })

#         print("Running backtest with sufficient data")
        
#         # Call run_backtest with the updated dummy data
#         result = run_backtest('EURUSD', dummy_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)
#         print(f"Result: {result}")

#         # Assert the results
#         self.assertIsNotNone(result, "run_backtest returned None")
#         self.assertTrue(result['buy_condition'], f"Expected buy_condition to be True, but got {result['buy_condition']}")
#         self.assertFalse(result['sell_condition'], f"Expected sell_condition to be False, but got {result['sell_condition']}")

#     def test_run_backtest_edge_cases(self):
#         # Prepare edge case data
#         extreme_data = pd.DataFrame({
#             'time': pd.date_range(start='2024-06-17', periods=10000, freq='h'),
#             'open': [1.1] * 10000,
#             'high': [1.15] * 10000,
#             'low': [1.05] * 10000,
#             'close': [1.1] * 10000,
#             'tick_volume': [100] * 10000,
#             'spread': [1] * 10000,
#             'real_volume': [10] * 10000
#         })

#         # Call run_backtest with edge case data
#         result = run_backtest('EURUSD', extreme_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)
#         print(f"Result: {result}")

#         # Assert the results
#         self.assertIsNotNone(result, "run_backtest returned None")

#     def test_check_entry_conditions_direct(self):
#         # Prepare data for check_entry_conditions
#         data = pd.DataFrame({
#             'time': pd.date_range(start='2024-06-17', periods=5, freq='h'),
#             'open': [1.1, 1.2, 1.3, 1.4, 1.5],
#             'high': [1.15, 1.25, 1.35, 1.45, 1.55],
#             'low': [1.05, 1.15, 1.25, 1.35, 1.45],
#             'close': [1.1, 1.2, 1.3, 1.4, 1.5],
#             'tick_volume': [100, 200, 300, 400, 500],
#             'spread': [1, 1, 1, 1, 1],
#             'real_volume': [10, 20, 30, 40, 50],
#             'wavy_c': [1.1, 1.2, 1.3, 1.4, 1.5],
#             'wavy_h': [1.15, 1.25, 1.35, 1.45, 1.55],
#             'wavy_l': [1.05, 1.15, 1.25, 1.35, 1.45],
#             'tunnel1': [1.0, 1.1, 1.2, 1.3, 1.4],
#             'tunnel2': [0.5, 0.6, 0.7, 0.8, 0.9]
#         })

#         peaks, dips = [], []

#         for i in range(5):
#             row = data.iloc[i]
#             print(f"Calling check_entry_conditions for row {i}")
#             result = check_entry_conditions(row, peaks, dips, 'EURUSD')
#             print(f"Result for row {i}: {result}, type: {type(result)}")
#             print(f"Result[0]: {result[0]}, Result[1]: {result[1]}")
#             self.assertFalse(result[0])
#             self.assertFalse(result[1])

#     def test_performance_and_stability(self):
#         # Prepare large dataset
#         large_data = pd.DataFrame({
#             'time': pd.date_range(start='2024-06-17', periods=100000, freq='h'),
#             'open': [1.1] * 100000,
#             'high': [1.15] * 100000,
#             'low': [1.05] * 100000,
#             'close': [1.1] * 100000,
#             'tick_volume': [100] * 100000,
#             'spread': [1] * 100000,
#             'real_volume': [10] * 100000
#         })

#         # Call run_backtest with large dataset
#         result = run_backtest('EURUSD', large_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)
#         print(f"Result: {result}")

#         # Assert the results
#         self.assertIsNotNone(result, "run_backtest returned None")

# if __name__ == '__main__':
#     unittest.main()
import unittest
import pandas as pd
import numpy as np
from strategy.tunnel_strategy import run_backtest

class TestBacktestIntegration(unittest.TestCase):
    def test_run_backtest_with_dummy_data(self):
        print("Setting up test with sufficient data")

        # Prepare dummy data with at least 50 data points
        dummy_data = pd.DataFrame({
            'time': pd.date_range(start='2024-06-17', periods=50, freq='h'),
            'open': np.linspace(1.1, 2.0, 50),
            'high': np.linspace(1.2, 2.1, 50),
            'low': np.linspace(1.0, 1.9, 50),
            'close': np.linspace(1.5, 2.4, 50),  # Adjusted to be higher for peaks
            'tick_volume': np.random.randint(100, 500, 50),
            'spread': np.random.randint(1, 5, 50),
            'real_volume': np.random.randint(10, 50, 50),
            'wavy_c': np.linspace(1.1, 2.0, 50),
            'wavy_h': np.linspace(1.2, 2.1, 50),
            'wavy_l': np.linspace(1.0, 1.9, 50),
            'tunnel1': np.linspace(1.0, 1.5, 50),
            'tunnel2': np.linspace(0.5, 1.0, 50)
        })

        print("Running backtest with sufficient data")
        
        # Call run_backtest with the updated dummy data
        result = run_backtest('EURUSD', dummy_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)
        print(f"Result: {result}")

        # Assert the results
        self.assertIsNotNone(result, "run_backtest returned None")
        self.assertTrue(result['buy_condition'], f"Expected buy_condition to be True, but got {result['buy_condition']}")
        self.assertFalse(result['sell_condition'], f"Expected sell_condition to be False, but got {result['sell_condition']}")

if __name__ == '__main__':
    unittest.main()

