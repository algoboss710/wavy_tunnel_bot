import unittest
from unittest.mock import patch
from datetime import datetime
from metatrader.data_retrieval import get_historical_data

class TestDataRetrieval(unittest.TestCase):
    @patch('MetaTrader5.copy_rates_range', return_value=[])
    def test_get_historical_data_success(self, mock_copy_rates_range):
        symbol = "EURUSD"
        timeframe = 1
        start_time = datetime(2023, 1, 1)
        end_time = datetime.now()
        data = get_historical_data(symbol, timeframe, start_time, end_time)
        self.assertIsNotNone(data)

    @patch('MetaTrader5.copy_rates_range', return_value=None)
    def test_get_historical_data_failure(self, mock_copy_rates_range):
        symbol = "EURUSD"
        timeframe = 1
        start_time = datetime(2023, 1, 1)
        end_time = datetime.now()
        data = get_historical_data(symbol, timeframe, start_time, end_time)
        self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()