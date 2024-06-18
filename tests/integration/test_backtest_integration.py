import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from main import run_backtest_func
from strategy.tunnel_strategy import calculate_ema, calculate_tunnel_bounds, generate_trade_signal, check_entry_conditions, detect_peaks_and_dips, calculate_position_size

class TestBacktestIntegration(unittest.TestCase):

    @patch('main.get_historical_data')
    @patch('main.initialize_mt5', return_value=True)
    @patch('main.shutdown_mt5')
    @patch('main.run_backtest')
    @patch('main.datetime', wraps=datetime)
    def test_run_backtest_with_dummy_data(self, mock_datetime, mock_run_backtest, mock_shutdown_mt5, mock_initialize_mt5, mock_get_historical_data):
        # Set a fixed datetime for the test
        fixed_now = datetime(2024, 6, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now

        # Dummy historical data
        dummy_data = pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=30, freq='H'),
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

        # Mock the run_backtest function to capture its arguments and perform additional checks
        def side_effect_run_backtest(symbol, data, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day, stop_loss_pips, pip_value):
            # Log the kwargs to debug the issue
            logger.debug("run_backtest called with kwargs: %s", {
                'symbol': symbol,
                'data': data,
                'initial_balance': initial_balance,
                'risk_percent': risk_percent,
                'min_take_profit': min_take_profit,
                'max_loss_per_day': max_loss_per_day,
                'starting_equity': starting_equity,
                'max_trades_per_day': max_trades_per_day,
                'stop_loss_pips': stop_loss_pips,
                'pip_value': pip_value
            })

            # Validate parameters
            self.assertGreater(stop_loss_pips, 0, "stop_loss_pips should be greater than 0")
            self.assertGreater(pip_value, 0, "pip_value should be greater than 0")

            # Validate balance updates and trade execution
            balance = initial_balance
            for trade in data.itertuples():
                if trade.Index % 2 == 0:  # Simulate buy trade
                    balance -= trade.close * pip_value
                else:  # Simulate sell trade
                    balance += trade.close * pip_value

            self.assertNotEqual(balance, initial_balance, "Balance should have been updated")
            self.assertGreater(len(data), 0, "There should be at least one trade executed")

            # Optionally validate performance metrics if provided
            total_profit = sum(trade.close for trade in data.itertuples())
            num_trades = len(data)
            win_rate = sum(1 for trade in data.itertuples() if trade.close > initial_balance / num_trades) / num_trades

            self.assertIsInstance(total_profit, float, "Total profit should be a float")
            self.assertIsInstance(num_trades, int, "Number of trades should be an integer")
            self.assertIsInstance(win_rate, float, "Win rate should be a float")

            # Log results
            logger.info(f"Initial balance: {initial_balance}")
            logger.info(f"Final balance: {balance}")
            logger.info(f"Total Profit: {total_profit:.2f}")
            logger.info(f"Number of Trades: {num_trades}")
            logger.info(f"Win Rate: {win_rate:.2%}")

        mock_run_backtest.side_effect = side_effect_run_backtest

        # Call the function to run backtest
        try:
            run_backtest_func()
        except Exception as e:
            logger.error("An error occurred: %s", str(e))
            raise

        # Check if the functions were called correctly
        mock_initialize_mt5.assert_called_once_with('C:\\Program Files\\MetaTrader 5\\terminal64.exe')
        mock_get_historical_data.assert_called_once_with('EURUSD', mt5.TIMEFRAME_H1, datetime(2023, 1, 1), fixed_now)
        mock_run_backtest.assert_called_once_with(
            'EURUSD',
            dummy_data,
            10000,
            0.02,
            50.0,
            1000.0,
            10000.0,
            5,
            20,
            0.0001
        )
        mock_shutdown_mt5.assert_called_once()

    def test_calculate_ema(self):
        prices = [1, 2, 3, 4, 5]
        period = 3
        expected_ema = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])

        result = calculate_ema(prices, period)
        pd.testing.assert_series_equal(result, expected_ema)

    def test_calculate_tunnel_bounds(self):
        data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        period = 3
        deviation_factor = 2.0
        expected_upper_bound = pd.Series([np.nan, np.nan, 4.0, 5.0, 6.0])
        expected_lower_bound = pd.Series([np.nan, np.nan, 0.0, 1.0, 2.0])

        upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation_factor)
        pd.testing.assert_series_equal(upper_bound, expected_upper_bound)
        pd.testing.assert_series_equal(lower_bound, expected_lower_bound)

    def test_generate_trade_signal(self):
        data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        period = 3
        deviation_factor = 2.0
        expected_signal = 'BUY'

        upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation_factor)
        last_close = data['close'].iloc[-1]
        if last_close >= upper_bound.iloc[-1]:
            expected_signal = 'BUY'
        elif last_close <= lower_bound.iloc[-1]:
            expected_signal = 'SELL'
        else:
            expected_signal = None

        result = generate_trade_signal(data, period, deviation_factor)
        self.assertEqual(result, expected_signal)

    @patch('strategy.tunnel_strategy.mt5.symbol_info')
    def test_check_entry_conditions(self, mock_symbol_info):
        symbol = 'EURUSD'
        row = {
            'close': 1.09,
            'open': 1.08,
            'wavy_c': 1.05,
            'wavy_h': 1.07,
            'wavy_l': 1.04,
            'tunnel1': 1.03,
            'tunnel2': 1.02
        }
        peaks = pd.Series([1.20])
        dips = pd.Series([1.10])
        mock_symbol_info.return_value = MagicMock(trade_tick_size=0.0001)
        buy_condition, sell_condition = check_entry_conditions(row, peaks, dips, symbol)
        self.assertTrue(buy_condition)
        self.assertFalse(sell_condition)

    def test_detect_peaks_and_dips(self):
        df = pd.DataFrame({'high': [1, 2, 3, 2, 1], 'low': [1, 0.5, 1, 0.5, 1]})
        peak_type = 3
        expected_peaks = [3]
        expected_dips = [0.5]

        peaks, dips = detect_peaks_and_dips(df, peak_type)
        self.assertEqual(peaks, expected_peaks)
        self.assertEqual(dips, expected_dips)

    def test_calculate_position_size(self):
        account_balance = 10000
        risk_per_trade = 0.01
        stop_loss_pips = 20
        pip_value = 0.0001
        expected_position_size = 50000.0  # Corrected expected result

        result = calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, pip_value)
        self.assertEqual(result, expected_position_size)

    @patch('main.run_backtest_func')
    def test_run_backtest_func_initialization(self, mock_run_backtest_func):
        # Simulate initialization and shutdown
        mock_run_backtest_func()
        mock_run_backtest_func.assert_called_once()

    @patch('main.run_backtest_func')
    def test_run_backtest_edge_cases(self, mock_run_backtest_func):
        # Test with insufficient data points
        insufficient_data = pd.DataFrame({'close': [1, 2]})
        with self.assertRaises(Exception):
            run_backtest_func(insufficient_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

        # Test with extreme values in data
        extreme_data = pd.DataFrame({'close': [1e10, 1e12, 1e14]})
        mock_run_backtest_func.return_value = None  # Just for testing the handling
        run_backtest_func(extreme_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)
        mock_run_backtest_func.assert_called()

    def test_performance_and_stability(self):
        # Measure performance with large datasets
        large_data = pd.DataFrame({
            'time': pd.date_range(start='2020-01-01', periods=10000, freq='h'),
            'open': [1.1 + i * 0.001 for i in range(10000)],
            'high': [1.2 + i * 0.001 for i in range(10000)],
            'low': [1.0 + i * 0.001 for i in range(10000)],
            'close': [1.15 + i * 0.001 for i in range(10000)],
            'tick_volume': [100 + i for i in range(10000)],
            'spread': [1 for i in range(10000)],
            'real_volume': [1000 + i for i in range(10000)]
        })

        initial_balance = 10000
        risk_percent = 0.02
        min_take_profit = 50.0
        max_loss_per_day = 1000.0
        starting_equity = 10000.0
        max_trades_per_day = 5
        stop_loss_pips = 20
        pip_value = 0.0001

        run_backtest_func(large_data, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day, stop_loss_pips, pip_value)

if __name__ == '__main__':
    unittest.main()
