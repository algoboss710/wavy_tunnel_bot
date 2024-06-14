import sys
import os

# Ensure the directory containing `config.py` is in the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
import pandas as pd
import logging
from datetime import datetime
from strategy.tunnel_strategy import calculate_ema, detect_peaks_and_dips, check_entry_conditions, manage_position, calculate_tunnel_bounds
from metatrader.data_retrieval import get_historical_data
from metatrader.trade_management import place_order, close_position
from utils.error_handling import handle_error
from metatrader.connection import initialize_mt5, shutdown_mt5  # Import these functions

def calculate_indicators(data):
    data['wavy_h'] = calculate_ema(data['high'], 34)
    data['wavy_c'] = calculate_ema(data['close'], 34)
    data['wavy_l'] = calculate_ema(data['low'], 34)
    data['tunnel1'] = calculate_ema(data['close'], 144)
    data['tunnel2'] = calculate_ema(data['close'], 169)
    data['long_term_ema'] = calculate_ema(data['close'], 200)
    return data

def run_backtest(symbol, data, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    balance = initial_balance
    trades = []
    stop_loss_pips = 20
    pip_value = Config.PIP_VALUE

    logging.info(f"Initial balance: {balance}")

    if stop_loss_pips == 0 or pip_value == 0:
        logging.error("stop_loss_pips or pip_value is zero. These values must not be zero.")
        return

    if data is not None and not data.empty:
        logging.info(f"Backtest data shape: {data.shape}")
        logging.info(f"Backtest data head:\n{data.head()}")
    else:
        logging.error(f"No historical data retrieved for {symbol} for backtesting")
        return

    if len(data) < 20:
        logging.error(f"Not enough data for symbol {symbol} to perform backtest")
        return

    data = calculate_indicators(data)
    peaks, dips = detect_peaks_and_dips(data, peak_type=21)

    for i in range(20, len(data)):
        logging.info(f"Iteration: {i}")
        
        row = data.iloc[i]
        buy_condition, sell_condition = check_entry_conditions(row, peaks, dips, symbol)

        if buy_condition:
            trade = {
                'entry_time': row['time'],
                'entry_price': row['close'],
                'volume': calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
            }
            trades.append(trade)
            execute_trade(trade, 'BUY')
            logging.info(f"Balance after BUY trade: {balance}")

        elif sell_condition:
            if trades:
                trade = trades[-1]
                trade['exit_time'] = row['time']
                trade['exit_price'] = row['close']
                trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['volume'] * pip_value
                balance += trade['profit']
                execute_trade(trade, 'SELL')
                logging.info(f"Balance after SELL trade: {balance}")

    logging.info(f"Final balance: {balance}")

    total_profit = sum(trade['profit'] for trade in trades if 'profit' in trade)
    num_trades = len(trades)
    win_rate = sum(1 for trade in trades if 'profit' in trade and trade['profit'] > 0) / num_trades if num_trades > 0 else 0
    max_drawdown = calculate_max_drawdown(trades, initial_balance)

    logging.info(f"Total Profit: {total_profit:.2f}")
    logging.info(f"Number of Trades: {num_trades}")
    logging.info(f"Win Rate: {win_rate:.2%}")
    logging.info(f"Maximum Drawdown: {max_drawdown:.2f}")

def calculate_max_drawdown(trades, initial_balance):
    peak_balance = initial_balance
    drawdown = 0
    max_drawdown = 0
    balance = initial_balance

    for trade in trades:
        if 'profit' in trade:
            balance += trade['profit']
            if balance > peak_balance:
                peak_balance = balance
            drawdown = peak_balance - balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return max_drawdown

def calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, pip_value):
    risk_amount = account_balance * risk_per_trade
    if stop_loss_pips == 0 or pip_value == 0:
        logging.error("stop_loss_pips or pip_value cannot be zero.")
        return 0
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return position_size

def execute_trade(trade, action):
    try:
        sl = trade['entry_price'] - (1.5 * Config.PIP_VALUE) if action == 'BUY' else trade['entry_price'] + (1.5 * Config.PIP_VALUE)
        tp = trade['entry_price'] + (2 * Config.PIP_VALUE) if action == 'BUY' else trade['entry_price'] - (2 * Config.PIP_VALUE)
        result = place_order(
            symbol=trade['symbol'],
            order_type=action.lower(),
            volume=trade['volume'],
            price=trade['entry_price'],
            sl=sl,
            tp=tp
        )
        if result != 'Order placed successfully':
            raise Exception("Failed to execute trade")
        return result
    except Exception as e:
        handle_error(e, "Failed to execute trade")
        return None

def main():
    try:
        logging.info("Initializing MetaTrader5...")
        if not initialize_mt5(Config.MT5_PATH):
            raise Exception("Failed to initialize MetaTrader5")
        logging.info("MetaTrader5 initialized successfully.")

        for symbol in Config.SYMBOLS:
            logging.info(f"Running backtest for {symbol}...")
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2022, 1, 5)
            initial_balance = Config.STARTING_EQUITY
            risk_percent = Config.RISK_PER_TRADE

            backtest_data = get_historical_data(symbol, Config.MT5_TIMEFRAME, start_date, end_date)
            if backtest_data is not None and not backtest_data.empty:
                logging.info(f"Backtest data shape: {backtest_data.shape}")
                logging.info(f"Backtest data head:\n{backtest_data.head()}")
            else:
                logging.error(f"No historical data retrieved for {symbol} for backtesting")
                continue

            if len(backtest_data) < 20:
                logging.error(f"Not enough data for symbol {symbol} to perform backtest")
                continue

            try:
                run_backtest(
                    symbol=symbol,
                    data=backtest_data,
                    initial_balance=initial_balance,
                    risk_percent=risk_percent,
                    min_take_profit=Config.MIN_TP_PROFIT,
                    max_loss_per_day=Config.MAX_LOSS_PER_DAY,
                    starting_equity=Config.STARTING_EQUITY,
                    max_trades_per_day=Config.LIMIT_NO_OF_TRADES
                )
                logging.info("Backtest completed successfully.")
            except Exception as e:
                handle_error(e, f"An error occurred during backtesting for {symbol}")

    except Exception as e:
        handle_error(e, "An error occurred in the backtest script")
        raise

    finally:
        logging.info("Shutting down MetaTrader5...")
        shutdown_mt5()
        logging.info("MetaTrader5 connection gracefully shut down.")

if __name__ == '__main__':
    main()
