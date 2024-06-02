import pandas as pd
from datetime import datetime
from metatrader.data_retrieval import get_historical_data
from strategy.tunnel_strategy import generate_trade_signal, execute_trade, manage_position
from utils.plotting import plot_backtest_results
from utils.error_handling import handle_error
from config import Config
import logging

def run_backtest(symbol, data, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    balance = initial_balance
    trades = []
    stop_loss_pips = 20  # Example value for stop loss in pips
    pip_value = Config.PIP_VALUE

    # Validate critical parameters
    if stop_loss_pips == 0:
        logging.error(f"stop_loss_pips is zero. This value must not be zero.")
        return
    if pip_value == 0:
        logging.error(f"pip_value is zero. This value must not be zero.")
        return

    for i in range(20, len(data)):  # Start after enough data points are available
        logging.info(f"Iteration: {i}")
        logging.info(f"Data shape: {data.iloc[:i+1].shape}")
        logging.info(f"Data head:\n{data.iloc[:i+1].head()}")

        # Calculate indicators and generate trading signals
        signal = generate_trade_signal(data.iloc[:i+1], period=20, deviation_factor=2.0)

        try:
            position_size = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
        except ZeroDivisionError as e:
            logging.error(f"Division by zero occurred in calculate_position_size: {e}. Variables - balance: {balance}, risk_percent: {risk_percent}, stop_loss_pips: {stop_loss_pips}, pip_value: {pip_value}")
            continue

        if signal == 'BUY':
            # Simulate trade entry
            trade = {
                'entry_time': data.iloc[i]['time'],
                'entry_price': data.iloc[i]['close'],
                'volume': position_size
            }
            trades.append(trade)
            execute_trade(trade)

        elif signal == 'SELL':
            # Simulate trade exit
            if trades:
                trade = trades[-1]
                trade['exit_time'] = data.iloc[i]['time']
                trade['exit_price'] = data.iloc[i]['close']
                trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['volume']
                balance += trade['profit']
                execute_trade(trade)

        manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

    total_profit = sum(trade['profit'] for trade in trades if 'profit' in trade)
    num_trades = len(trades)
    win_rate = sum(1 for trade in trades if 'profit' in trade and trade['profit'] > 0) / num_trades
    max_drawdown = calculate_max_drawdown(trades, initial_balance)

    logging.info(f"Total Profit: {total_profit:.2f}")
    logging.info(f"Number of Trades: {num_trades}")
    logging.info(f"Win Rate: {win_rate:.2%}")
    logging.info(f"Maximum Drawdown: {max_drawdown:.2f}")
    print(f"stop_loss_pips: {stop_loss_pips}")
    print(f"pip_value: {pip_value}")
    # Plot backtest results
    plot_backtest_results(data, trades)

def calculate_max_drawdown(trades, initial_balance):
    balance = initial_balance
    max_balance = initial_balance
    max_drawdown = 0

    for trade in trades:
        balance += trade['profit']
        max_balance = max(max_balance, balance)
        drawdown = max_balance - balance
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


def calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, pip_value):
    risk_amount = account_balance * risk_per_trade
    if stop_loss_pips == 0 or pip_value == 0:
        logging.error("stop_loss_pips or pip_value cannot be zero.")
        return 0  # Return 0 or handle the error appropriately

    position_size = risk_amount / (stop_loss_pips * pip_value)
    return position_size

def generate_trade_signal(data, period, deviation_factor):
    # Placeholder for generating trade signals
    return 'BUY' if data['close'].iloc[-1] > data['close'].mean() else 'SELL'

def execute_trade(trade):
    # Placeholder for executing trades
    pass

def manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    # Placeholder for managing positions
    pass
