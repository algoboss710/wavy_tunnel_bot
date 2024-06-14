# tests/backtesting/test_backtest.py
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from backtesting.backtest import run_backtest, calculate_max_drawdown, calculate_position_size

class TestBacktest(unittest.TestCase):

    def setUp(self):
        # Create a mock historical data
        self.mock_data = pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=30, freq='H'),
            'open': [i + 1 for i in range(30)],
            'high': [i + 2 for i in range(30)],
            'low': [i for i in range(30)],
            'close': [i + 1.5 for i in range(30)],
            'tick_volume': [100 + i for i in range(30)],
            'spread': [1 for i in range(30)],
            'real_volume': [1000 + i for i in range(30)]
        })

    @patch('backtesting.backtest.get_historical_data')
    @patch('backtesting.backtest.generate_trade_signal', return_value='BUY')
    @patch('backtesting.backtest.execute_trade')
    @patch('backtesting.backtest.manage_position')
    def test_run_backtest(self, mock_manage_position, mock_execute_trade, mock_generate_trade_signal, mock_get_historical_data):
        # Mock the get_historical_data function to return the mock data
        mock_get_historical_data.return_value = self.mock_data

        run_backtest(
            symbol='EURUSD',
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=50,
            max_loss_per_day=1000,
            starting_equity=10000,
            max_trades_per_day=5
        )

        mock_get_historical_data.assert_called_once()
        mock_generate_trade_signal.assert_called()
        mock_execute_trade.assert_called()
        mock_manage_position.assert_called()
    
    def test_calculate_max_drawdown(self):
        trades = [
            {'profit': 100},
            {'profit': -50},
            {'profit': 200},
            {'profit': -100}
        ]
        initial_balance = 1000
        max_drawdown = calculate_max_drawdown(trades, initial_balance)
        self.assertEqual(max_drawdown, 100)

    def test_calculate_position_size(self):
        account_balance = 10000
        risk_per_trade = 0.01
        stop_loss_pips = 50
        pip_value = 10
        expected_position_size = 0.2
        position_size = calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, pip_value)
        self.assertEqual(position_size, expected_position_size)

if __name__ == '__main__':
    unittest.main()
