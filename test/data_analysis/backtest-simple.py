import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
from collections import defaultdict
import json
import os
import concurrent.futures

# Create test_logs folder if it doesn't exist
os.makedirs('test_logs_backtest', exist_ok=True)

# Set up logging for successful trades
success_logger = logging.getLogger('successful_trades')
success_logger.setLevel(logging.INFO)
success_handler = logging.FileHandler('test_logs_backtest/successful_trades.log')
success_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
success_logger.addHandler(success_handler)

# Set up logging for failed trades
failure_logger = logging.getLogger('failed_trades')
failure_logger.setLevel(logging.ERROR)
failure_handler = logging.FileHandler('test_logs_backtest/failed_trades.log')
failure_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
failure_logger.addHandler(failure_handler)

# Set up summary logging
summary_logger = logging.getLogger('summary')
summary_logger.setLevel(logging.INFO)
summary_handler = logging.FileHandler('test_logs_backtest/summary.log')
summary_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
summary_logger.addHandler(summary_handler)

# Global variables to track trades and performance
trade_summary = defaultdict(lambda: {"analyzed": 0, "executed_long": 0, "executed_short": 0, "profit_long": 0, "profit_short": 0})
initial_balance = 10000  # Starting balance for backtesting
balance = initial_balance

# Connect to MetaTrader 5
if not mt5.initialize():
    logging.error("MetaTrader5 initialization failed")
    mt5.shutdown()
    quit()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_historical_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        logging.warning(f"Failed to retrieve data for {symbol} on {timeframe} timeframe")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    logging.info(f"Retrieved {len(df)} bars for {symbol} on {timeframe} timeframe")
    return df

def wavy_tunnel_strategy(df):
    df['wavy_h'] = calculate_ema(df['high'], 34)
    df['wavy_c'] = calculate_ema(df['close'], 34)
    df['wavy_l'] = calculate_ema(df['low'], 34)
    df['tunnel1'] = calculate_ema(df['close'], 144)
    df['tunnel2'] = calculate_ema(df['close'], 169)
    df['rsi'] = calculate_rsi(df)

    max_wavy = df[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)
    min_wavy = df[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)
    max_tunnel = df[['tunnel1', 'tunnel2']].max(axis=1)
    min_tunnel = df[['tunnel1', 'tunnel2']].min(axis=1)

    rsi_upper = 70
    rsi_lower = 30

    df['long_condition'] = (df['open'] > max_wavy) & (min_wavy > max_tunnel) & (df['rsi'] < rsi_upper)
    df['short_condition'] = (df['open'] < min_wavy) & (max_wavy < min_tunnel) & (df['rsi'] > rsi_lower)
    df['exit_long'] = df['close'] < min_wavy
    df['exit_short'] = df['close'] > max_wavy

    return df

def simulate_trade(entry_price, exit_price, order_type, lot_size=0.01):
    global balance
    profit = 0
    if order_type == 'buy':
        profit = (exit_price - entry_price) * lot_size * 100000  # Adjust for lot size and pip value
    elif order_type == 'sell':
        profit = (entry_price - exit_price) * lot_size * 100000

    balance += profit
    return profit

def backtest(symbol, timeframe, start_date, end_date, lot_size=0.01):
    global balance
    logger = logging.getLogger(f'{symbol}_{timeframe}')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'test_logs_backtest/{symbol}_{timeframe}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    df = get_historical_data(symbol, timeframe, start_date, end_date)
    if df is None:
        logger.warning(f"No data for {symbol} on {timeframe} timeframe, skipping backtest")
        return

    df = wavy_tunnel_strategy(df)
    logger.info(f"Starting backtest for {symbol} on {timeframe} timeframe")

    position = None
    trades_analyzed = 0
    trades_executed_long = 0
    trades_executed_short = 0
    profit_long = 0
    profit_short = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        trades_analyzed += 1

        if position is None:
            # Entry conditions
            if row['long_condition']:
                logger.info(f"Opening long position at {row['open']} on {row['time']}")
                position = {'type': 'buy', 'entry_price': row['open'], 'entry_time': row['time']}
                trades_executed_long += 1
            elif row['short_condition']:
                logger.info(f"Opening short position at {row['open']} on {row['time']}")
                position = {'type': 'sell', 'entry_price': row['open'], 'entry_time': row['time']}
                trades_executed_short += 1
        else:
            # Exit conditions
            if position['type'] == 'buy' and row['exit_long']:
                profit = simulate_trade(position['entry_price'], row['close'], 'buy', lot_size)
                logger.info(f"Closing long position at {row['close']} on {row['time']} | Profit: {profit}")
                profit_long += profit
                position = None
            elif position['type'] == 'sell' and row['exit_short']:
                profit = simulate_trade(position['entry_price'], row['close'], 'sell', lot_size)
                logger.info(f"Closing short position at {row['close']} on {row['time']} | Profit: {profit}")
                profit_short += profit
                position = None

    # Log summary of trades for this symbol and timeframe
    summary_logger.info(f"Summary for {symbol} on {timeframe}:")
    summary_logger.info(f"Trades analyzed: {trades_analyzed}")
    summary_logger.info(f"Trades executed (long): {trades_executed_long}")
    summary_logger.info(f"Trades executed (short): {trades_executed_short}")
    summary_logger.info(f"Total profit (long): {profit_long}")
    summary_logger.info(f"Total profit (short): {profit_short}")
    summary_logger.info(f"Net Profit/Loss: {profit_long + profit_short}")
    summary_logger.info("------------------------")

    # Update global trade summary
    trade_summary[f"{symbol}_{timeframe}"]["analyzed"] += trades_analyzed
    trade_summary[f"{symbol}_{timeframe}"]["executed_long"] += trades_executed_long
    trade_summary[f"{symbol}_{timeframe}"]["executed_short"] += trades_executed_short
    trade_summary[f"{symbol}_{timeframe}"]["profit_long"] += profit_long
    trade_summary[f"{symbol}_{timeframe}"]["profit_short"] += profit_short

    logger.info(f"Final balance for {symbol} on {timeframe}: ${balance:.2f}")
    logger.info(f"Backtest complete for {symbol} on {timeframe}")

def run_backtests(symbols, timeframes, start_date, end_date):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(symbols) * len(timeframes)) as executor:
            futures = []
            for symbol in symbols:
                for tf in timeframes:
                    futures.append(executor.submit(backtest, symbol, tf, start_date, end_date))

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

            logging.info("All backtests complete")

            # Log total summary across all pairs and timeframes
            summary_logger.info("Overall Summary:")
            total_analyzed = 0
            total_executed_long = 0
            total_executed_short = 0
            total_profit_long = 0
            total_profit_short = 0

            for key, stats in trade_summary.items():
                summary_logger.info(f"Summary for {key}:")
                summary_logger.info(f"Trades analyzed: {stats['analyzed']}")
                summary_logger.info(f"Trades executed (long): {stats['executed_long']}")
                summary_logger.info(f"Trades executed (short): {stats['executed_short']}")
                summary_logger.info(f"Total profit (long): {stats['profit_long']}")
                summary_logger.info(f"Total profit (short): {stats['profit_short']}")
                summary_logger.info(f"Net Profit/Loss: {stats['profit_long'] + stats['profit_short']}")
                summary_logger.info("------------------------")

                total_analyzed += stats['analyzed']
                total_executed_long += stats['executed_long']
                total_executed_short += stats['executed_short']
                total_profit_long += stats['profit_long']
                total_profit_short += stats['profit_short']

            # Final summary for all trades
            summary_logger.info("Total Summary across all pairs and timeframes:")
            summary_logger.info(f"Total trades analyzed: {total_analyzed}")
            summary_logger.info(f"Total trades executed (long): {total_executed_long}")
            summary_logger.info(f"Total trades executed (short): {total_executed_short}")
            summary_logger.info(f"Total profit (long): {total_profit_long}")
            summary_logger.info(f"Total profit (short): {total_profit_short}")
            summary_logger.info(f"Net Profit/Loss: {total_profit_long + total_profit_short}")
            summary_logger.info("------------------------")

    except KeyboardInterrupt:
        logging.info("Backtests interrupted by user")
    finally:
        logging.info(f"Final simulated balance: ${balance:.2f}")
        mt5.shutdown()

if __name__ == "__main__":
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    timeframes = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    logging.info("Starting Wavy Tunnel Strategy Backtest")
    logging.info(f"Testing on symbols: {symbols}")
    logging.info(f"Timeframes: {timeframes}")
    logging.info(f"Date range: {start_date.date()} to {end_date.date()}")

    run_backtests(symbols, timeframes, start_date, end_date)
