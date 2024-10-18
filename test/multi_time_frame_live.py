import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(filename='multi_timeframe_wavy_tunnel_strategy_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables to track trades and performance
trade_summary = defaultdict(int)
successful_trades = []
initial_balance = 0

# Initialize connection to MetaTrader 5
if not mt5.initialize():
    logging.error("MetaTrader 5 initialization failed")
    mt5.shutdown()
    quit()

def get_historical_data(symbol, timeframe, num_bars):
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    logging.info(f"Retrieved {len(df)} bars for {symbol} on {timeframe} timeframe")
    return df

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def wavy_tunnel_signals(data):
    data['wavy_h'] = calculate_ema(data['high'], 34)
    data['wavy_c'] = calculate_ema(data['close'], 34)
    data['wavy_l'] = calculate_ema(data['low'], 34)
    data['tunnel1'] = calculate_ema(data['close'], 144)
    data['tunnel2'] = calculate_ema(data['close'], 169)

    data['long_condition'] = (data['open'] > data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)) & \
                             (data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1) > data[['tunnel1', 'tunnel2']].max(axis=1))

    data['short_condition'] = (data['open'] < data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)) & \
                              (data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1) < data[['tunnel1', 'tunnel2']].min(axis=1))

    return data

def initialize_metrics():
    global initial_balance
    account_info = mt5.account_info()
    if account_info is not None:
        initial_balance = account_info.balance
    else:
        logging.error("Failed to retrieve initial account balance")

def update_trade_summary(action, result):
    trade_summary[f"{action}_{result}"] += 1

def log_successful_trade(symbol, timeframe, action, entry_price, exit_price, profit):
    trade_details = {
        "symbol": symbol,
        "timeframe": timeframe,
        "action": action,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "profit": profit
    }
    successful_trades.append(trade_details)

    with open("multi_timeframe_wavy_tunnel_successful_trades.log", "a") as f:
        f.write(json.dumps(trade_details) + "\n")

def log_position(action, symbol, timeframes, entry_price, position_type):
    log_entry = {
        "time": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "timeframes": [str(tf) for tf in timeframes],
        "entry_price": entry_price,
        "position_type": position_type
    }

    with open("multi_timeframe_wavy_tunnel_positions.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def print_performance_metrics():
    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to retrieve final account info")
        return

    final_balance = account_info.balance
    total_profit = final_balance - initial_balance

    logging.info("Performance Metrics:")
    logging.info(f"Initial Balance: ${initial_balance:.2f}")
    logging.info(f"Final Balance: ${final_balance:.2f}")
    logging.info(f"Total Profit: ${total_profit:.2f}")
    logging.info(f"Total Trades Attempted: {sum(trade_summary.values())}")
    logging.info(f"Successful Trades: {len(successful_trades)}")
    logging.info("Trade Summary:")
    for action, count in trade_summary.items():
        logging.info(f"  {action}: {count}")

def run_multi_timeframe_strategy(symbol, timeframes):
    data = {}
    signals = {}
    position = None
    entry_price = None

    # Use the shortest timeframe for trade execution
    shortest_tf = min(timeframes)

    while True:
        for tf in timeframes:
            data[tf] = get_historical_data(symbol, tf, 1000)
            signals[tf] = wavy_tunnel_signals(data[tf])

        latest_bar = signals[shortest_tf].iloc[-1]

        # Check if signals align across all timeframes
        long_signals = all(signals[tf]['long_condition'].iloc[-1] for tf in timeframes)
        short_signals = all(signals[tf]['short_condition'].iloc[-1] for tf in timeframes)

        logging.info(f"Time: {latest_bar['time']}, Close: {latest_bar['close']:.5f}, Position: {position}")
        logging.info(f"Long signals aligned: {long_signals}, Short signals aligned: {short_signals}")

        if position is None:
            if long_signals:
                position = 'Long'
                entry_price = latest_bar['close']
                logging.info(f"Opened Long position at {entry_price:.5f}")
                update_trade_summary("open_long", "success")
                log_position("open", symbol, timeframes, entry_price, "Long")
            elif short_signals:
                position = 'Short'
                entry_price = latest_bar['close']
                logging.info(f"Opened Short position at {entry_price:.5f}")
                update_trade_summary("open_short", "success")
                log_position("open", symbol, timeframes, entry_price, "Short")
        else:
            if position == 'Long':
                if latest_bar['close'] < latest_bar['wavy_l']:
                    profit = latest_bar['close'] - entry_price
                    logging.info(f"Closed Long position. Entry: {entry_price:.5f}, Exit: {latest_bar['close']:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, shortest_tf, "Close Long", entry_price, latest_bar['close'], profit)
                    update_trade_summary("close_long", "success")
                    log_position("close", symbol, timeframes, latest_bar['close'], "Long")
                    position = None
            elif position == 'Short':
                if latest_bar['close'] > latest_bar['wavy_h']:
                    profit = entry_price - latest_bar['close']
                    logging.info(f"Closed Short position. Entry: {entry_price:.5f}, Exit: {latest_bar['close']:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, shortest_tf, "Close Short", entry_price, latest_bar['close'], profit)
                    update_trade_summary("close_short", "success")
                    log_position("close", symbol, timeframes, latest_bar['close'], "Short")
                    position = None

        logging.info("------------------------")

        # Wait for the next tick
        time.sleep(shortest_tf)

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]

    logging.info(f"Starting Multi-Timeframe Wavy Tunnel Strategy on {symbol}")
    logging.info(f"Timeframes: {timeframes}")

    initialize_metrics()

    try:
        run_multi_timeframe_strategy(symbol, timeframes)
    except KeyboardInterrupt:
        logging.info("Strategy stopped by user")
    finally:
        print_performance_metrics()
        mt5.shutdown()
        logging.info("MetaTrader 5 connection closed")