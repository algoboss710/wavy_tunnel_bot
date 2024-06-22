# import unittest
# import pandas as pd
# import numpy as np
# from unittest.mock import patch, MagicMock
# from backtesting.backtest import run_backtest
# from strategy.tunnel_strategy import check_entry_conditions

# class TestBacktestIntegration(unittest.TestCase):

#     @patch('backtesting.backtest.calculate_ema')
#     @patch('backtesting.backtest.detect_peaks_and_dips')
#     @patch('backtesting.backtest.check_entry_conditions')
#     @patch('backtesting.backtest.mt5.symbol_info')
#     # Remove the @patch for run_backtest to call it directly
#     def test_run_backtest_with_dummy_data(self, mock_symbol_info, mock_check_entry_conditions, mock_detect_peaks_and_dips, mock_calculate_ema):
#         # Prepare dummy data
#         dummy_data = pd.DataFrame({
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
#             'wavy_l': [1.05, 1.15, 1.25, 1.35, 1.45]
#         })

#         print("Mocking return values")
#         mock_calculate_ema.return_value = pd.Series([1.1, 1.2, 1.3, 1.4, 1.5])
#         mock_detect_peaks_and_dips.return_value = ([], [])
#         mock_check_entry_conditions.return_value = (True, False)
#         mock_symbol_info.return_value = MagicMock(trade_tick_size=0.0001)

#         # Call run_backtest with dummy data
#         print("Calling run_backtest directly")
#         result = run_backtest('EURUSD', dummy_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)
#         print(f"Result: {result}")

#         # Assert the results
#         self.assertIsNotNone(result, "run_backtest returned None")
#         self.assertTrue(result['buy_condition'])
#         self.assertFalse(result['sell_condition'])

#     @patch('backtesting.backtest.run_backtest')
#     @patch('backtesting.backtest.mt5.symbol_info')
#     def test_run_backtest_edge_cases(self, mock_symbol_info, mock_run_backtest):
#         # Prepare edge case data
#         extreme_data = pd.DataFrame({
#             'time': pd.date_range(start='2024-06-17', periods=10000, freq='h'),
#             'open': [1.1] * 10000,
#             'high': [1.15] * 10000,
#             'low': [1.05] * 10000,
#             'close': [1.1] * 10000,
#             'tick_volume': [100] * 10000,
#             'spread': [1] * 10000,
#             'real_volume': [10] * 10000,
#             'wavy_c': [1.1] * 10000,
#             'wavy_h': [1.15] * 10000,
#             'wavy_l': [1.05] * 10000
#         })

#         # Mock symbol info
#         mock_symbol_info.return_value = MagicMock(trade_tick_size=0.0001)

#         run_backtest('EURUSD', extreme_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

#         mock_run_backtest.assert_called_once_with('EURUSD', extreme_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

#     @patch('strategy.tunnel_strategy.mt5.symbol_info')
#     def test_check_entry_conditions_direct(self, mock_symbol_info):
#         # Mock symbol info
#         mock_symbol_info.return_value = MagicMock(trade_tick_size=0.0001)

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
#             'wavy_l': [1.05, 1.15, 1.25, 1.35, 1.45]
#         })

#         peaks, dips = [], []

#         for i in range(5):
#             row = data.iloc[i]
#             print(f"Calling check_entry_conditions for row {i}")
#             result = check_entry_conditions(row, peaks, dips, 'EURUSD')
#             print(f"Result for row {i}: {result}, type: {type(result)}")
#             print(f"Result[0]: {result[0]}, Result[1]: {result[1]}")
#             # Since the current test data and logic return False, we assert it correctly.
#             self.assertFalse(result[0])  # Asserting the boolean value directly
#             self.assertFalse(result[1])  # Asserting the boolean value directly

#     @patch('backtesting.backtest.run_backtest')
#     @patch('backtesting.backtest.mt5.symbol_info')
#     def test_performance_and_stability(self, mock_symbol_info, mock_run_backtest):
#         # Prepare large dataset
#         large_data = pd.DataFrame({
#             'time': pd.date_range(start='2024-06-17', periods=100000, freq='h'),
#             'open': [1.1] * 100000,
#             'high': [1.15] * 100000,
#             'low': [1.05] * 100000,
#             'close': [1.1] * 100000,
#             'tick_volume': [100] * 100000,
#             'spread': [1] * 100000,
#             'real_volume': [10] * 100000,
#             'wavy_c': [1.1] * 100000,
#             'wavy_h': [1.15] * 100000,
#             'wavy_l': [1.05] * 100000
#         })

#         # Mock symbol info
#         mock_symbol_info.return_value = MagicMock(trade_tick_size=0.0001)

#         run_backtest('EURUSD', large_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

#         mock_run_backtest.assert_called_once_with('EURUSD', large_data, 10000, 0.02, 50.0, 1000.0, 10000.0, 5, 20, 0.0001)

# if __name__ == '__main__':
#     unittest.main()
# backtesting/backtest.py
import logging
import pandas as pd
import numpy as np
from strategy.tunnel_strategy import check_entry_conditions, generate_trade_signal, manage_position, calculate_position_size, detect_peaks_and_dips
from metatrader.indicators import calculate_ema
from metatrader.trade_management import execute_trade

def calculate_max_drawdown(trades, initial_balance):
    balance = initial_balance
    max_balance = initial_balance
    max_drawdown = 0

    for trade in trades:
        if 'profit' in trade:
            balance += trade['profit']
            max_balance = max(max_balance, balance)
            drawdown = max_balance - balance
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown

def run_backtest(symbol, data, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day, stop_loss_pips, pip_value):
    balance = initial_balance
    trades = []

    logging.info(f"Initial balance: {balance}")
    print(f"Initial balance: {balance}")

    # Validate critical parameters
    if stop_loss_pips == 0:
        logging.error(f"stop_loss_pips is zero. This value must not be zero.")
        return
    if pip_value == 0:
        logging.error(f"pip_value is zero. This value must not be zero.")
        return

    peak_type = 21

    # Calculate indicators and peaks/dips for the entire dataset
    data['wavy_h'] = calculate_ema(data['high'], 34)
    data['wavy_c'] = calculate_ema(data['close'], 34)
    data['wavy_l'] = calculate_ema(data['low'], 34)
    data['tunnel1'] = calculate_ema(data['close'], 144)
    data['tunnel2'] = calculate_ema(data['close'], 169)
    data['long_term_ema'] = calculate_ema(data['close'], 200)
    peaks, dips = detect_peaks_and_dips(data, peak_type)

    buy_condition = False
    sell_condition = False

    for i in range(20, len(data)):  # Start after enough data points are available
        logging.info(f"Iteration: {i}")
        logging.info(f"Data shape: {data.iloc[:i+1].shape}")
        logging.info(f"Data head:\n{data.iloc[:i+1].head()}")

        print(f"Iteration: {i}")
        print(f"Data shape: {data.iloc[:i+1].shape}")
        print(f"Data head:\n{data.iloc[:i+1].head()}")

        # Generate trading signals
        signal = generate_trade_signal(data.iloc[:i+1], period=20, deviation_factor=2.0)
        print(f"Generated signal: {signal}")

        try:
            position_size = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
        except ZeroDivisionError as e:
            logging.error(f"Division by zero occurred in calculate_position_size: {e}. Variables - balance: {balance}, risk_percent: {risk_percent}, stop_loss_pips: {stop_loss_pips}, pip_value: {pip_value}")
            continue

        row = data.iloc[i]
        print(f"Calling check_entry_conditions for row {i}")
        logging.info(f"Calling check_entry_conditions for row {i}")
        buy_condition, sell_condition = check_entry_conditions(row, peaks, dips, symbol)
        print(f"Result for check_entry_conditions at row {i}: buy_condition={buy_condition}, sell_condition={sell_condition}")
        logging.info(f"Result for check_entry_conditions at row {i}: buy_condition={buy_condition}, sell_condition={sell_condition}")

        if buy_condition:
            print(f"Buy condition met at row {i}.")
            logging.info(f"Buy condition met at row {i}.")
            # Simulate trade entry
            trade = {
                'entry_time': data.iloc[i]['time'],
                'entry_price': data.iloc[i]['close'],
                'volume': position_size,
                'symbol': symbol,
                'action': 'BUY',
                'sl': data.iloc[i]['close'] - (1.5 * data['close'].rolling(window=20).std().iloc[i]),
                'tp': data.iloc[i]['close'] + (2 * data['close'].rolling(window=20).std().iloc[i])
            }
            trades.append(trade)
            execute_trade(trade)
            logging.info(f"Balance after BUY trade: {balance}")

        elif sell_condition:
            print(f"Sell condition met at row {i}.")
            logging.info(f"Sell condition met at row {i}.")
            # Simulate trade exit
            if trades:
                trade = trades[-1]
                trade['exit_time'] = data.iloc[i]['time']
                trade['exit_price'] = data.iloc[i]['close']
                trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['volume'] * pip_value
                try:
                    balance += trade['profit']
                except KeyError as e:
                    logging.error(f"KeyError occurred while updating balance: {e}")
                execute_trade(trade)
                logging.info(f"Balance after SELL trade: {balance}")

        manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

    logging.info(f"Final balance: {balance}")
    print(f"Final balance: {balance}")

    total_profit = sum(trade['profit'] for trade in trades if 'profit' in trade)
    num_trades = len(trades)
    win_rate = sum(1 for trade in trades if 'profit' in trade and trade['profit'] > 0) / num_trades if num_trades > 0 else 0
    max_drawdown = calculate_max_drawdown(trades, initial_balance)

    logging.info(f"Total Profit: {total_profit:.2f}")
    logging.info(f"Number of Trades: {num_trades}")
    logging.info(f"Win Rate: {win_rate:.2%}")
    logging.info(f"Maximum Drawdown: {max_drawdown:.2f}")

    print(f"Total Profit: {total_profit:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}")

    return {
        'total_profit': total_profit,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'buy_condition': buy_condition,
        'sell_condition': sell_condition
    }
