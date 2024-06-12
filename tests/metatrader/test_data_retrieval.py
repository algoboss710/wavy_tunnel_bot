import unittest
from unittest.mock import patch
from datetime import datetime
from metatrader.data_retrieval import get_historical_data
import MetaTrader5 as mt5
import pandas as pd

class TestDataRetrieval(unittest.TestCase):
    @patch('MetaTrader5.copy_rates_range', return_value=[
        {'time': 1633072800, 'open': 1.1600, 'high': 1.1700, 'low': 1.1500, 'close': 1.1650, 'tick_volume': 100, 'spread': 1, 'real_volume': 1000}
    ])
    def test_get_historical_data_valid_data(self, mock_copy_rates_range):
        symbol = "EURUSD"
        timeframe = mt5.TIMEFRAME_M1
        start_time = datetime(2023, 1, 1)
        end_time = datetime.now()
        data = get_historical_data(symbol, timeframe, start_time, end_time)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('time', data.columns)
        self.assertIn('open', data.columns)

    @patch('MetaTrader5.copy_rates_range', return_value=[])
    def test_get_historical_data_empty(self, mock_copy_rates_range):
        symbol = "EURUSD"
        timeframe = mt5.TIMEFRAME_M1
        start_time = datetime(2023, 1, 1)
        end_time = datetime.now()
        data = get_historical_data(symbol, timeframe, start_time, end_time)
        self.assertIsNone(data)

    @patch('MetaTrader5.copy_rates_range', side_effect=Exception("MT5 error"))
    def test_get_historical_data_exception(self, mock_copy_rates_range):
        symbol = "EURUSD"
        timeframe = mt5.TIMEFRAME_M1
        start_time = datetime(2023, 1, 1)
        end_time = datetime.now()
        data = get_historical_data(symbol, timeframe, start_time, end_time)
        self.assertIsNone(data)

    @patch('MetaTrader5.copy_rates_range', return_value=None)
    def test_get_historical_data_none(self, mock_copy_rates_range):
        symbol = "INVALID_SYMBOL"
        timeframe = mt5.TIMEFRAME_M1
        start_time = datetime(2023, 1, 1)
        end_time = datetime.now()
        data = get_historical_data(symbol, timeframe, start_time, end_time)
        self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()
