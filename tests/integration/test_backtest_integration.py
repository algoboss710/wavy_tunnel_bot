import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from backtesting.backtest import run_backtest
from strategy.tunnel_strategy import check_entry_conditions

class TestBacktestIntegration(unittest.TestCase):

    @patch('backtesting.backtest.calculate_ema')
    @patch('backtesting.backtest.detect_peaks_and_dips')
    @patch('backtesting.backtest.check_entry_conditions')
    @patch('backtesting.backtest.mt5.symbol_info')
    def test_run_backtest_with_dummy_data(self, mock_symbol_info, mock_check_entry_conditions, mock_detect_peaks_and_dips, mock_calculate_ema):
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

        mock_calculate_ema.return_value = pd.Series([1.1, 1.2, 1.3, 1.4, 1.5])
        mock_detect_peaks_and_dips.return_value = ([], [])
        mock_check_entry_conditions.return_value = (True, False)

        # Mock symbol info
        mock_symbol_info.return_value = MagicMock(trade_tick_size=0.0001)

        # Call run_backtest with dummy data
        result = run_backtest('EURUSD', dummy_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

        # Assert the results
        self.assertTrue(result['buy_condition'])
        self.assertFalse(result['sell_condition'])

    @patch('backtesting.backtest.run_backtest')
    @patch('backtesting.backtest.mt5.symbol_info')
    def test_run_backtest_edge_cases(self, mock_symbol_info, mock_run_backtest):
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

        # Mock symbol info
        mock_symbol_info.return_value = MagicMock(trade_tick_size=0.0001)

        run_backtest('EURUSD', extreme_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

        mock_run_backtest.assert_called_once_with('EURUSD', extreme_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

    @patch('backtesting.backtest.run_backtest')
    @patch('backtesting.backtest.mt5.symbol_info')
    def test_check_entry_conditions(self, mock_symbol_info, mock_run_backtest):
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
            result = check_entry_conditions(row, peaks, dips, 'EURUSD')
            self.assertTrue(result['buy_condition'])

    @patch('backtesting.backtest.run_backtest')
    @patch('backtesting.backtest.mt5.symbol_info')
    def test_performance_and_stability(self, mock_symbol_info, mock_run_backtest):
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

        # Mock symbol info
        mock_symbol_info.return_value = MagicMock(trade_tick_size=0.0001)

        run_backtest('EURUSD', large_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

        mock_run_backtest.assert_called_once_with('EURUSD', large_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

if __name__ == '__main__':
    unittest.main()
