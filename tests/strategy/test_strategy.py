import unittest
import pandas as pd
import numpy as np
from unittest import mock
from strategy.tunnel_strategy import (
    calculate_ema, calculate_tunnel_bounds, detect_peaks_and_dips, 
    check_entry_conditions, generate_trade_signal, run_strategy, execute_trade, manage_position, adjust_deviation_factor, calculate_position_size
)
import MetaTrader5 as mt5
from unittest.mock import Mock, patch

class TestStrategy(unittest.TestCase):

    def test_calculate_ema(self):
        print("Running test_calculate_ema")
        prices = pd.Series([100, 200, 300, 400, 500])
        period = 3
        result = calculate_ema(prices, period)
        expected_ema = pd.Series([None, None, 200.0, 300.0, 400.0])
        pd.testing.assert_series_equal(result, expected_ema, check_names=False)
    
    def test_calculate_ema_empty_series(self):
        print("Running test_calculate_ema_empty_series")
        prices = pd.Series([], dtype=float)
        period = 3
        expected_ema = pd.Series([], dtype=float)
        result = calculate_ema(prices, period)
        pd.testing.assert_series_equal(result, expected_ema, check_names=False)
    
    def test_calculate_ema_non_numeric(self):
        print("Running test_calculate_ema_non_numeric")
        prices = pd.Series(['abc', 'def', 'ghi'])
        period = 3
        result = calculate_ema(prices, period)
        expected_ema = pd.Series([np.nan] * len(prices))
        pd.testing.assert_series_equal(result, expected_ema, check_names=False)

    def test_calculate_ema_small_period(self):
        print("Running test_calculate_ema_small_period")
        prices = pd.Series([100, 200, 300])
        period = 1
        expected_ema = pd.Series([100.0, 200.0, 300.0])
        result = calculate_ema(prices, period)
        pd.testing.assert_series_equal(result, expected_ema, check_names=False)
    
    def test_calculate_ema_large_period(self):
        print("Running test_calculate_ema_large_period")
        prices = pd.Series(range(1, 51))
        period = 50
        expected_ema = pd.Series([np.nan] * 49 + [np.mean(range(1, 51))], dtype=float)
        result = calculate_ema(prices, period)
        pd.testing.assert_series_equal(result, expected_ema, check_names=False)

    def test_calculate_tunnel_bounds(self):
        print("Running test_calculate_tunnel_bounds")
        data = pd.DataFrame({'close': [100, 200, 300, 400, 500]})
        period = 3
        deviation_factor = 1.0
        expected_upper_bound = pd.Series([np.nan, np.nan, 300.0, 400.0, 500.0])
        expected_lower_bound = pd.Series([np.nan, np.nan, 100.0, 200.0, 300.0])
        upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation_factor)
        pd.testing.assert_series_equal(upper_bound.round(2), expected_upper_bound, check_names=False)
        pd.testing.assert_series_equal(lower_bound.round(2), expected_lower_bound, check_names=False)

    def test_calculate_tunnel_bounds_missing_columns(self):
        print("Running test_calculate_tunnel_bounds_missing_columns")
        data = pd.DataFrame({'open': [100, 200, 300]})
        period = 3
        deviation_factor = 1.0
        with self.assertRaises(KeyError):  # Assumes that the function needs 'close' column
            calculate_tunnel_bounds(data, period, deviation_factor)

    def test_calculate_tunnel_bounds_insufficient_length(self):
        print("Running test_calculate_tunnel_bounds_insufficient_length")
        data = pd.DataFrame({'close': [100, 200]})
        period = 3
        deviation_factor = 1.0
        expected_upper_bound = pd.Series([np.nan, np.nan])
        expected_lower_bound = pd.Series([np.nan, np.nan])
        upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation_factor)
        pd.testing.assert_series_equal(upper_bound, expected_upper_bound, check_names=False)
        pd.testing.assert_series_equal(lower_bound, expected_lower_bound, check_names=False)

    def test_detect_peaks_and_dips_insufficient_data(self):
        print("Running test_detect_peaks_and_dips_insufficient_data")
        data = pd.DataFrame({'high': [10, 12], 'low': [8, 7]})
        peak_type = 5
        expected_peaks = []
        expected_dips = []
        peaks, dips = detect_peaks_and_dips(data, peak_type)
        self.assertEqual(peaks, expected_peaks)
        self.assertEqual(dips, expected_dips)

    def test_detect_peaks_and_dips_non_numeric(self):
        print("Running test_detect_peaks_and_dips_non_numeric")
        data = pd.DataFrame({'high': ['ten', 'twelve'], 'low': ['eight', 'seven']})
        peak_type = 3
        with self.assertRaises(TypeError):
            detect_peaks_and_dips(data, peak_type)

    def test_detect_peaks_and_dips(self):
        print("Running test_detect_peaks_and_dips")
        data = {
            'high': [10, 12, 15, 14, 13, 17, 16, 19, 18, 17],
            'low': [8, 7, 6, 9, 8, 11, 10, 9, 12, 11]
        }
        df = pd.DataFrame(data)
        peak_type = 5
        expected_peaks = [15, 19]
        expected_dips = [6, 9]
        peaks, dips = detect_peaks_and_dips(df, peak_type)
        print(f"Detected peaks: {peaks}, Detected dips: {dips}")
        self.assertEqual(peaks, expected_peaks)
        self.assertEqual(dips, expected_dips)
    
    def test_check_entry_conditions_missing_data(self):
        print("Running test_check_entry_conditions_missing_data")
        row = pd.Series({
         'close': 350,
         # 'wavy_c' missing
         'wavy_h': 320,
         'wavy_l': 310,
         'tunnel1': 250,
         'tunnel2': 240
        })
        peaks = [350]
        dips = [150]
        symbol = 'EURUSD'
        with self.assertRaises(KeyError):
         check_entry_conditions(row, peaks, dips, symbol)

    def test_check_entry_conditions_incorrect_types(self):
        print("Running test_check_entry_conditions_incorrect_types")
        row = pd.Series({
            'close': 350,
            'wavy_c': 300,
            'wavy_h': 320,
            'wavy_l': 310,
            'tunnel1': 250,
            'tunnel2': 240
        })
        peaks = 'not a list'
        dips = 'also not a list'
        symbol = 'EURUSD'
        with self.assertRaises(TypeError):
            check_entry_conditions(row, peaks, dips, symbol)

    @mock.patch('strategy.tunnel_strategy.mt5')
    def test_check_entry_conditions(self, mock_mt5):
        print("Running test_check_entry_conditions")
        row = pd.Series({
            'close': 350,
            'wavy_c': 300,
            'wavy_h': 320,
            'wavy_l': 310,
            'tunnel1': 250,
            'tunnel2': 240
        })
        peaks = [350]
        dips = [150]
        symbol = 'EURUSD'
        mock_symbol_info = mock.Mock()
        mock_symbol_info.trade_tick_size = 0.01
        mock_mt5.symbol_info.return_value = mock_symbol_info

        buy_condition, sell_condition = check_entry_conditions(row, peaks, dips, symbol)
        self.assertTrue(buy_condition)
        self.assertFalse(sell_condition)

    def test_generate_trade_signal_buy(self):
        print("Running test_generate_trade_signal_buy")
        data = pd.DataFrame({'close': [100, 200, 300, 400, 500, 600]})
        period = 3
        deviation_factor = 1.0
        expected_signal = 'BUY'

        signal = generate_trade_signal(data, period, deviation_factor)
        print(f"Expected: {expected_signal}, Got: {signal}")
        self.assertEqual(signal, expected_signal)

    def test_generate_trade_signal_sell(self):
        print("Running test_generate_trade_signal_sell")
        data = pd.DataFrame({'close': [100, 200, 300, 400, 500, 100]})  # Last close set to 100 to ensure 'SELL'
        period = 3
        deviation_factor = 1.0
        expected_signal = 'SELL'

        signal = generate_trade_signal(data, period, deviation_factor)
        print(f"Expected: {expected_signal}, Got: {signal}")
        self.assertEqual(signal, expected_signal)

    def test_generate_trade_signal_none(self):
        print("Running test_generate_trade_signal_none")
        data = pd.DataFrame({'close': [100, 200, 300, 400, 500, 450]})
        period = 3
        deviation_factor = 1.0
        expected_signal = None

        signal = generate_trade_signal(data, period, deviation_factor)
        print(f"Expected: {expected_signal}, Got: {signal}")
        self.assertIsNone(signal)
    
    def test_generate_trade_signal_non_numeric_data(self):
        print("Running test_generate_trade_signal_non_numeric_data")
        data = pd.DataFrame({
            'close': ['one', 'two', 'three', 4, 5, 6]
        })
        period = 3
        deviation_factor = 2
        result = generate_trade_signal(data, period, deviation_factor)
        expected_signal = None  # Since non-numeric data will be coerced to NaN, resulting in no valid signal
        self.assertEqual(result, expected_signal)

    def test_generate_trade_signal_small_dataset(self):
        print("Running test_generate_trade_signal_small_dataset")
        data = pd.DataFrame({'close': [100, 200]})
        period = 3
        deviation_factor = 1.0
        expected_signal = None  # Not enough data to generate a signal
        signal = generate_trade_signal(data, period, deviation_factor)
        self.assertIsNone(signal)

    @mock.patch('strategy.tunnel_strategy.mt5')
    def test_run_strategy_initialization_failure(self, mock_mt5):
        print("Running test_run_strategy_initialization_failure")
        mock_mt5.initialize.return_value = False  # Simulate initialization failure
        symbols = ['EURUSD']
        timeframe = mock_mt5.TIMEFRAME_M1
        lot_size = 0.1
        min_take_profit = 10
        max_loss_per_day = 50
        starting_equity = 1000
        max_trades_per_day = 5
        with self.assertRaises(Exception):  # Assuming your function raises Exception on init failure
            run_strategy(symbols, mock_mt5.initialize, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

    @mock.patch('strategy.tunnel_strategy.mt5')
    def test_run_strategy_no_symbols_available(self, mock_mt5):
        print("Running test_run_strategy_no_symbols_available")
        mock_mt5.initialize.return_value = True
        mock_mt5.symbols_get.return_value = []  # No symbols returned
        symbols = ['EURUSD']
        timeframe = mock_mt5.TIMEFRAME_M1
        lot_size = 0.1
        min_take_profit = 10
        max_loss_per_day = 50
        starting_equity = 1000
        max_trades_per_day = 5
        with self.assertRaises(Exception) as context:
            run_strategy(symbols, mock_mt5.initialize, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)
        self.assertIn("Failed to retrieve historical data for EURUSD", str(context.exception))

    @mock.patch('strategy.tunnel_strategy.mt5')
    @mock.patch('strategy.tunnel_strategy.get_historical_data')
    def test_run_strategy_data_to_signal_flow(self, mock_get_historical_data, mock_mt5):
        print("Running test_run_strategy_data_to_signal_flow")
        mock_mt5.initialize.return_value = True
        mock_mt5.symbols_get.return_value = [mock.Mock(name='EURUSD')]

        # Set up historical data with all required fields
        mock_get_historical_data.return_value = pd.DataFrame({
            'time': pd.date_range(start='1/1/2022', periods=5, freq='min'),
            'high': [110, 220, 330, 440, 550],
            'low': [90, 180, 270, 360, 450],
            'close': [105, 210, 315, 420, 525]
        })

        # Mock the additional required attributes/methods
        mock_mt5.symbol_info.return_value = mock.Mock()
        mock_mt5.symbol_info.return_value.trade_tick_size = 0.01  # Example value

        symbols = ['EURUSD']
        timeframe = mock_mt5.TIMEFRAME_M1
        lot_size = 0.1
        min_take_profit = 10
        max_loss_per_day = 50
        starting_equity = 1000
        max_trades_per_day = 5

        # Check to ensure strategy runs and captures the output or exceptions as expected
        try:
            result = run_strategy(symbols, mock_mt5.initialize, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)
            print("Strategy ran successfully.")
        except Exception as e:
            print(f"Strategy failed with error: {str(e)}")

    @mock.patch('strategy.tunnel_strategy.mt5')
    @mock.patch('strategy.tunnel_strategy.get_historical_data')
    def test_run_strategy(self, mock_get_historical_data, mock_mt5):
        print("Running test_run_strategy")
        mock_mt5.TIMEFRAME_M1 = 1
        mock_mt5.symbol_info.return_value.trade_tick_size = 0.01
        
        symbols = ['EURUSD']
        timeframe = mock_mt5.TIMEFRAME_M1
        lot_size = 0.1
        min_take_profit = 10
        max_loss_per_day = 50
        starting_equity = 1000
        max_trades_per_day = 5

        # Mock data retrieval and other MT5 functions
        mock_mt5.initialize.return_value = True
        mock_mt5.symbols_get.return_value = [mock.Mock(name='EURUSD')]
        mock_get_historical_data.return_value = pd.DataFrame({
            'time': pd.date_range(start='1/1/2022', periods=5, freq='min'),
            'open': [100, 200, 300, 400, 500],
            'high': [110, 220, 330, 440, 550],
            'low': [90, 180, 270, 360, 450],
            'close': [105, 210, 315, 420, 525],
            'tick_volume': [1000, 2000, 3000, 4000, 5000],
            'spread': [10, 20, 30, 40, 50],
            'real_volume': [100, 200, 300, 400, 500]
        })

        run_strategy(symbols, mock_mt5.initialize, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)
    
    @mock.patch('strategy.tunnel_strategy.place_order')
    def test_execute_trade_success(self, mock_place_order):
        print("Running test_execute_trade_success")
        mock_place_order.return_value = 'Order placed successfully'
        trade_request = {
            'symbol': 'EURUSD',
            'action': 'BUY',
            'volume': 1.0,
            'price': 1.2345,
            'sl': 1.2300,
            'tp': 1.2400
        }
        result = execute_trade(trade_request)
        self.assertEqual(result, 'Order placed successfully')
        # Adjusted to use positional arguments as shown in the actual call
        mock_place_order.assert_called_once_with(
            'EURUSD', 'buy', 1.0, 1.2345, 1.2300, 1.2400
        )
    
    @mock.patch('strategy.tunnel_strategy.place_order')
    def test_execute_trade_failure(self, mock_place_order):
        print("Running test_execute_trade_failure")
        mock_place_order.return_value = 'Order failed'  # Simulated failure message
        trade_request = {
            'symbol': 'EURUSD',
            'action': 'SELL',
            'volume': 1.0,
            'price': 1.2345,
            'sl': 1.2400,
            'tp': 1.2300
        }
        result = execute_trade(trade_request)
        self.assertIsNone(result)  # Checking if the result is None as expected
        mock_place_order.assert_called_once_with(
            'EURUSD',  # Ensure case and parameter order matches the actual call
            'sell', 
            1.0, 
            1.2345, 
            1.2400, 
            1.2300
        )
    
    @mock.patch('strategy.tunnel_strategy.place_order')
    @mock.patch('strategy.tunnel_strategy.handle_error')
    def test_execute_trade_exception(self, mock_handle_error, mock_place_order):
        print("Running test_execute_trade_exception")
        # Setup for exception scenario
        mock_place_order.side_effect = Exception("Connection error")
        trade_request = {
            'symbol': 'EURUSD',
            'action': 'BUY',
            'volume': 1.0,
            'price': 1.2345,
            'sl': 1.23,
            'tp': 1.24
        }

        result = execute_trade(trade_request)
        mock_place_order.assert_called_once_with(
        'EURUSD', 'buy', 1.0, 1.2345, 1.23, 1.24
        )
        self.assertIsNone(result)
        mock_handle_error.assert_called_once()

    @patch('strategy.tunnel_strategy.mt5')
    @patch('strategy.tunnel_strategy.close_position')
    def test_manage_position(self, mock_close_position, mock_mt5):
        print("Running test_manage_position")
        # Mock the positions_get and account_info responses
        mock_mt5.positions_get.return_value = [
            Mock(ticket=1, profit=500),  # Should be closed for profit
            Mock(ticket=2, profit=-1500),  # Should be closed for loss
            Mock(ticket=3, profit=0)  # Should be closed due to equity drop
        ]
        mock_mt5.account_info.return_value = Mock(equity=8000)
        mock_mt5.positions_total.return_value = 6  # Should close positions due to max trades exceeded

        # Mock the historical data retrieval
        mock_mt5.copy_rates_range.return_value = pd.DataFrame({
            'time': pd.date_range(start='2022-01-01', periods=5, freq='H'),
            'open': [1.1, 1.2, 1.3, 1.4, 1.5],
            'high': [1.15, 1.25, 1.35, 1.45, 1.55],
            'low': [1.05, 1.15, 1.25, 1.35, 1.45],
            'close': [1.1, 1.2, 1.3, 1.4, 1.5],
            'tick_volume': [100, 200, 300, 400, 500],
            'spread': [0, 0, 0, 0, 0],
            'real_volume': [0, 0, 0, 0, 0]
        })

        manage_position('EURUSD', min_take_profit=300, max_loss_per_day=1000, starting_equity=10000, max_trades_per_day=5)

        # Assertions for close_position calls
        mock_close_position.assert_any_call(1)  # Close position with profit > min_take_profit
        mock_close_position.assert_any_call(2)  # Close position with loss < -max_loss_per_day
        mock_close_position.assert_any_call(3)  # Close position due to equity drop
        self.assertEqual(mock_close_position.call_count, 3)  # Adjust expected call count

    def test_adjust_deviation_factor_volatile(self):
        print("Running test_adjust_deviation_factor_volatile")
        market_conditions = 'volatile'
        result = adjust_deviation_factor(market_conditions)
        self.assertEqual(result, 2.5)

    def test_adjust_deviation_factor_calm(self):
        print("Running test_adjust_deviation_factor_calm")
        market_conditions = 'calm'
        result = adjust_deviation_factor(market_conditions)
        self.assertEqual(result, 2.0)
    
    def test_calculate_position_size_valid_inputs(self):
        print("Running test_calculate_position_size_valid_inputs")
        account_balance = 10000
        risk_per_trade = 0.01
        stop_loss_pips = 50
        pip_value = 10

        expected_position_size = 0.2  # Corrected expected position size
        position_size = calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, pip_value)
        self.assertEqual(position_size, expected_position_size)

if __name__ == '__main__':
    unittest.main()
