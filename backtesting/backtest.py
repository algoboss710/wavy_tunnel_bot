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

    print(f"Peaks: {peaks}")
    print(f"Dips: {dips}")

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
        print(f"Row {i} data: {row}")
        print(f"Peaks: {peaks}")
        print(f"Dips: {dips}")
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
