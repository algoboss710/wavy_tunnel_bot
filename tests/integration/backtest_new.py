import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from backtest import run_backtest  # Adjust the import according to the actual module path

class TestRunBacktest(unittest.TestCase):

    def setUp(self):
        # Prepare sample market data
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='h'),  # Use 'h' instead of 'H'
            'open': pd.Series([1.1000 + i*0.0001 for i in range(100)]),
            'high': pd.Series([1.1005 + i*0.0001 for i in range(100)]),
            'low': pd.Series([1.0995 + i*0.0001 for i in range(100)]),
            'close': pd.Series([1.1002 + i*0.0001 for i in range(100)])
        }
        self.test_data = pd.DataFrame(data)
        self.stop_loss_pips = 10
        self.pip_value = 0.1
        self.initial_balance = 10000
        self.risk_percent = 1
        self.min_take_profit = 50
        self.max_loss_per_day = 500
        self.starting_equity = 10000
        self.max_trades_per_day = 5

    @patch('backtest.calculate_ema')
    @patch('backtest.detect_peaks_and_dips')
    @patch('backtest.generate_trade_signal')
    @patch('backtest.calculate_position_size')
    @patch('backtest.check_entry_conditions')
    @patch('backtest.execute_trade')
    @patch('backtest.calculate_max_drawdown')
    def test_run_backtest(self, mock_calculate_max_drawdown, mock_execute_trade, mock_check_entry_conditions, mock_calculate_position_size, mock_generate_trade_signal, mock_detect_peaks_and_dips, mock_calculate_ema):
        # Mock method return values
        mock_calculate_ema.return_value = self.test_data['close']
        mock_detect_peaks_and_dips.return_value = (pd.Series([0]*100), pd.Series([0]*100))
        mock_generate_trade_signal.return_value = [None]*100
        mock_calculate_position_size.return_value = 1.0
        mock_check_entry_conditions.return_value = (False, False)
        mock_execute_trade.return_value = None
        mock_calculate_max_drawdown.return_value = -0.05

        # Run the backtest
        results = run_backtest(
            symbol='EURUSD',
            data=self.test_data,
            initial_balance=self.initial_balance,
            risk_percent=self.risk_percent,
            min_take_profit=self.min_take_profit,
            max_loss_per_day=self.max_loss_per_day,
            starting_equity=self.starting_equity,
            max_trades_per_day=self.max_trades_per_day,
            stop_loss_pips=self.stop_loss_pips,
            pip_value=self.pip_value
        )

        # Assertions
        mock_calculate_ema.assert_called()
        mock_detect_peaks_and_dips.assert_called()
        mock_generate_trade_signal.assert_called()
        mock_calculate_position_size.assert_called()
        mock_check_entry_conditions.assert_called()
        mock_execute_trade.assert_called()
        mock_calculate_max_drawdown.assert_called()

        # Validate logs and performance metrics (example)
        self.assertIn('total_profit', results)
        self.assertIn('num_trades', results)
        self.assertIn('win_rate', results)
        self.assertIn('max_drawdown', results)
        self.assertGreaterEqual(results['num_trades'], 0)

if __name__ == '__main__':
    unittest.main()
