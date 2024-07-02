# import logging
# import pandas as pd
# import numpy as np

# def calculate_max_drawdown(trades, initial_balance):
#     balance = initial_balance
#     max_balance = initial_balance
#     max_drawdown = 0

#     for trade in trades:
#         if 'profit' in trade:
#             balance += trade['profit']
#             max_balance = max(max_balance, balance)
#             drawdown = max_balance - balance
#             max_drawdown = max(max_drawdown, drawdown)

#     return max_drawdown

# def run_backtest(symbol, data, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, stop_loss_pips, pip_value, max_trades_per_day=None):
#     from strategy.tunnel_strategy import generate_trade_signal, manage_position, calculate_position_size, detect_peaks_and_dips
#     from metatrader.indicators import calculate_ema
#     from metatrader.trade_management import execute_trade

#     balance = initial_balance
#     trades = []
#     trades_today = 0
#     current_day = data.iloc[0]['time'].date()
#     max_drawdown = 0
#     daily_loss = 0

#     logging.info(f"Initial balance: {balance}")
#     print(f"Initial balance: {balance}")

#     # Validate critical parameters
#     if stop_loss_pips == 0 or pip_value == 0:
#         raise ZeroDivisionError("stop_loss_pips and pip_value must not be zero.")

#     peak_type = 21

#     # Calculate indicators and peaks/dips for the entire dataset
#     data['wavy_h'] = calculate_ema(data['high'], 34)
#     data['wavy_c'] = calculate_ema(data['close'], 34)
#     data['wavy_l'] = calculate_ema(data['low'], 34)
#     data['tunnel1'] = calculate_ema(data['close'], 144)
#     data['tunnel2'] = calculate_ema(data['close'], 169)
#     data['long_term_ema'] = calculate_ema(data['close'], 200)

#     peaks, dips = detect_peaks_and_dips(data, peak_type)

#     for i in range(20, len(data)):  # Start after enough data points are available
#         logging.info(f"Iteration: {i}, trades_today: {trades_today}, current_day: {current_day}")

#         # Check if it's a new day
#         if data.iloc[i]['time'].date() != current_day:
#             logging.info(f"New day detected: {data.iloc[i]['time'].date()}, resetting trades_today and daily_loss.")
#             current_day = data.iloc[i]['time'].date()
#             trades_today = 0
#             daily_loss = 0

#         if max_trades_per_day is not None and trades_today >= max_trades_per_day:
#             logging.info(f"Max trades per day reached at row {i}.")
#             continue

#         # Generate trading signals
#         buy_condition, sell_condition = generate_trade_signal(data.iloc[:i+1], period=20, deviation_factor=2.0)
#         if buy_condition is None or sell_condition is None:
#             continue

#         try:
#             position_size = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
#         except ZeroDivisionError as e:
#             logging.error(f"Division by zero occurred in calculate_position_size: {e}. Variables - balance: {balance}, risk_percent: {risk_percent}, stop_loss_pips: {stop_loss_pips}, pip_value: {pip_value}")
#             continue

#         row = data.iloc[i]

#         if buy_condition and (max_trades_per_day is None or trades_today < max_trades_per_day):
#             logging.info(f"Buy condition met at row {i}.")
#             trade = {
#                 'entry_time': data.iloc[i]['time'],
#                 'entry_price': data.iloc[i]['close'],
#                 'volume': position_size,
#                 'symbol': symbol,
#                 'action': 'BUY',
#                 'sl': data.iloc[i]['close'] - (1.5 * data['close'].rolling(window=20).std().iloc[i]),
#                 'tp': data.iloc[i]['close'] + (2 * data['close'].rolling(window=20).std().iloc[i])
#             }
#             trades.append(trade)
#             trades_today += 1
#             execute_trade(trade)
#             logging.info(f"Balance after BUY trade: {balance}")

#         elif sell_condition and (max_trades_per_day is None or trades_today < max_trades_per_day):
#             logging.info(f"Sell condition met at row {i}.")
#             if trades:
#                 trade = trades[-1]
#                 trade['exit_time'] = data.iloc[i]['time']
#                 trade['exit_price'] = data.iloc[i]['close']
#                 trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['volume'] * pip_value
#                 balance += trade['profit']
#                 execute_trade(trade)
#                 trades_today += 1
#                 logging.info(f"Balance after SELL trade: {balance}")

#         manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

#     logging.info(f"Final balance: {balance}")
#     print(f"Final balance: {balance}")

#     total_profit = sum(trade['profit'] for trade in trades if 'profit' in trade)
#     num_trades = len(trades)
#     win_rate = sum(1 for trade in trades if 'profit' in trade and trade['profit'] > 0) / num_trades if num_trades > 0 else 0
#     max_drawdown = calculate_max_drawdown(trades, initial_balance)

#     logging.info(f"Total Profit: {total_profit:.2f}")
#     logging.info(f"Number of Trades: {num_trades}")
#     logging.info(f"Win Rate: {win_rate:.2%}")
#     logging.info(f"Maximum Drawdown: {max_drawdown:.2f}")

#     print(f"Total Profit: {total_profit:.2f}")
#     print(f"Number of Trades: {num_trades}")
#     print(f"Win Rate: {win_rate:.2%}")
#     print(f"Maximum Drawdown: {max_drawdown:.2f}")

#     return {
#         'total_profit': total_profit,
#         'num_trades': num_trades,
#         'win_rate': win_rate,
#         'max_drawdown': max_drawdown,
#         'buy_condition': buy_condition,
#         'sell_condition': sell_condition,
#         'trades': trades
#     }

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
import logging
import pandas as pd
import numpy as np
import cProfile
import pstats
from io import StringIO

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

def run_backtest(symbol, data, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, stop_loss_pips, pip_value, max_trades_per_day=None, slippage=0, transaction_cost=0):
    from strategy.tunnel_strategy import generate_trade_signal, manage_position, calculate_position_size, detect_peaks_and_dips
    from metatrader.indicators import calculate_ema
    from metatrader.trade_management import execute_trade

    # Profiling setup
    pr = cProfile.Profile()
    pr.enable()

    try:
        balance = initial_balance
        trades = []
        trades_today = 0
        current_day = data.iloc[0]['time'].date()
        max_drawdown = 0
        daily_loss = 0
        buy_condition = False  # Initialize buy_condition
        sell_condition = False  # Initialize sell_condition

        logging.info(f"Initial balance: {balance}")
        print(f"Initial balance: {balance}")

        # Validate critical parameters
        if stop_loss_pips <= 0 or pip_value <= 0:
            raise ZeroDivisionError("stop_loss_pips and pip_value must be greater than zero.")

        peak_type = 21

        # Calculate indicators and peaks/dips for the entire dataset
        data.loc[:, 'wavy_h'] = calculate_ema(data['high'], 34)
        data.loc[:, 'wavy_c'] = calculate_ema(data['close'], 34)
        data.loc[:, 'wavy_l'] = calculate_ema(data['low'], 34)
        data.loc[:, 'tunnel1'] = calculate_ema(data['close'], 144)
        data.loc[:, 'tunnel2'] = calculate_ema(data['close'], 169)
        data.loc[:, 'long_term_ema'] = calculate_ema(data['close'], 200)

        peaks, dips = detect_peaks_and_dips(data, peak_type)

        for i in range(34, len(data)):  # Start after enough data points are available
            logging.info(f"Iteration: {i}, trades_today: {trades_today}, current_day: {current_day}")

            # Check if it's a new day
            if data.iloc[i]['time'].date() != current_day:
                logging.info(f"New day detected: {data.iloc[i]['time'].date()}, resetting trades_today and daily_loss.")
                current_day = data.iloc[i]['time'].date()
                trades_today = 0
                daily_loss = 0

            if max_trades_per_day is not None and trades_today >= max_trades_per_day:
                logging.info(f"Max trades per day reached at row {i}.")
                continue

            # Generate trading signals
            buy_condition, sell_condition = generate_trade_signal(data.iloc[:i+1], period=20, deviation_factor=2.0)
            if buy_condition is None or sell_condition is None:
                continue

            try:
                position_size = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
            except ZeroDivisionError as e:
                logging.error(f"Division by zero occurred in calculate_position_size: {e}. Variables - balance: {balance}, risk_percent: {risk_percent}, stop_loss_pips: {stop_loss_pips}, pip_value: {pip_value}")
                continue

            row = data.iloc[i]

            if buy_condition and (max_trades_per_day is None or trades_today < max_trades_per_day):
                logging.info(f"Buy condition met at row {i}.")
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
                trades_today += 1
                execute_trade(trade)
                logging.info(f"Balance after BUY trade: {balance}")

            elif sell_condition and (max_trades_per_day is None or trades_today < max_trades_per_day):
                logging.info(f"Sell condition met at row {i}.")
                if trades:
                    trade = trades[-1]
                    trade['exit_time'] = data.iloc[i]['time']
                    trade['exit_price'] = data.iloc[i]['close']
                    trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['volume'] * pip_value
                    balance += trade['profit']
                    execute_trade(trade)
                    trades_today += 1
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
            'sell_condition': sell_condition,
            'trades': trades,
            'total_slippage_costs': len(trades) * slippage,
            'total_transaction_costs': len(trades) * transaction_cost
        }

    finally:
        pr.disable()

        # Output profiling results
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        print(s.getvalue())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
