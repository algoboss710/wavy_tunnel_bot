import unittest
from unittest.mock import patch
from strategy.tunnel_strategy import run_strategy

class TestStrategy(unittest.TestCase):
    @patch('metatrader.data_retrieval.get_historical_data', return_value=None)
    def test_run_strategy_failure(self, mock_get_historical_data):
        symbols = ["EURUSD"]
        mt5_init = True
        timeframe = 1
        lot_size = 0.1
        min_take_profit = 100
        max_loss_per_day = 1000
        starting_equity = 10000
        max_traders_per_day = 5
        with self.assertRaises(Exception):
            run_strategy(symbols, mt5_init, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_traders_per_day)

if __name__ == '__main__':
    unittest.main()