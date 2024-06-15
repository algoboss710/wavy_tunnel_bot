import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from main import run_backtest_func

class TestBacktestIntegration(unittest.TestCase):

    @patch('main.initialize_mt5', return_value=True)
    @patch('main.shutdown_mt5')
    @patch('main.get_historical_data')
    @patch('main.run_backtest')
    @patch('main.datetime', wraps=datetime)
    def test_run_backtest_with_dummy_data(self, mock_datetime, mock_run_backtest, mock_get_historical_data, mock_shutdown_mt5, mock_initialize_mt5):
        # Set a fixed datetime for the test
        fixed_now = datetime(2024, 6, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now

        # Dummy historical data
        dummy_data = pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=30, freq='h'),
            'open': [1.1 + i * 0.001 for i in range(30)],
            'high': [1.2 + i * 0.001 for i in range(30)],
            'low': [1.0 + i * 0.001 for i in range(30)],
            'close': [1.15 + i * 0.001 for i in range(30)],
            'tick_volume': [100 + i for i in range(30)],
            'spread': [1 for i in range(30)],
            'real_volume': [1000 + i for i in range(30)]
        })

        # Configure the mock to return the dummy data
        mock_get_historical_data.return_value = dummy_data

        # Call the function to run backtest
        run_backtest_func()

        # Check if the functions were called correctly
        mock_initialize_mt5.assert_called_once_with('C:\\Program Files\\MetaTrader 5\\terminal64.exe')
        mock_get_historical_data.assert_called_once_with('EURUSD', mt5.TIMEFRAME_H1, datetime(2023, 1, 1), fixed_now)
        mock_run_backtest.assert_called_once()
        mock_shutdown_mt5.assert_called_once()

if __name__ == '__main__':
    unittest.main()
