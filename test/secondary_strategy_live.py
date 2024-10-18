import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(filename='wavy_tunnel_strategy_second_log.txt', level=logging.INFO,
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

def wavy_tunnel_strategy(data):
    data['wavy_h'] = calculate_ema(data['high'], 34)
    data['wavy_c'] = calculate_ema(data['close'], 34)
    data['wavy_l'] = calculate_ema(data['low'], 34)
    data['tunnel1'] = calculate_ema(data['close'], 144)
    data['tunnel2'] = calculate_ema(data['close'], 169)

    # Primary strategy conditions
    data['primary_long'] = (data['open'] > data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)) & \
                           (data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1) > data[['tunnel1', 'tunnel2']].max(axis=1))
    data['primary_short'] = (data['open'] < data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)) & \
                            (data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1) < data[['tunnel1', 'tunnel2']].min(axis=1))

    # Secondary strategy conditions
    data['secondary_long'] = (data['close'] > data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)) & \
                             (data['close'] < data[['tunnel1', 'tunnel2']].min(axis=1))
    data['secondary_short'] = (data['close'] < data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)) & \
                              (data['close'] > data[['tunnel1', 'tunnel2']].max(axis=1))

    return data

def calculate_zone_percentage(price, wavy_level, tunnel_level):
    return abs(price - wavy_level) / abs(tunnel_level - wavy_level)

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

    with open("wavy_tunnel_secondary_successful_trades.log", "a") as f:
        f.write(json.dumps(trade_details) + "\n")

def log_position(action, symbol, timeframe, entry_price, strategy):
    log_entry = {
        "time": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "timeframe": str(timeframe),
        "entry_price": entry_price,
        "strategy": strategy
    }

    with open("wavy_tunnel_secondary_positions.log", "a") as f:
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

def run_combined_strategy(symbol, timeframe, max_zone_entry=0.25):
    position = None
    entry_price = None

    while True:
        data = get_historical_data(symbol, timeframe, 1000)
        data = wavy_tunnel_strategy(data)

        latest_bar = data.iloc[-1]

        logging.info(f"Evaluating {symbol} on {timeframe} timeframe at {latest_bar['time']}")
        logging.info(f"Current price: Open: {latest_bar['open']:.5f}, Close: {latest_bar['close']:.5f}")

        if position is None:
            if latest_bar['primary_long']:
                position = 'Long'
                entry_price = latest_bar['close']
                logging.info(f"Primary Strategy: Opened Long position at {entry_price:.5f}")
                update_trade_summary("open_long_primary", "success")
                log_position("open", symbol, timeframe, entry_price, "Primary Long")
            elif latest_bar['primary_short']:
                position = 'Short'
                entry_price = latest_bar['close']
                logging.info(f"Primary Strategy: Opened Short position at {entry_price:.5f}")
                update_trade_summary("open_short_primary", "success")
                log_position("open", symbol, timeframe, entry_price, "Primary Short")
            elif latest_bar['secondary_long']:
                zone_percentage = calculate_zone_percentage(latest_bar['close'], latest_bar['wavy_h'], latest_bar['tunnel1'])
                if zone_percentage <= max_zone_entry:
                    position = 'Long'
                    entry_price = latest_bar['close']
                    logging.info(f"Secondary Strategy: Opened Long position at {entry_price:.5f}, Zone entry: {zone_percentage:.2%}")
                    update_trade_summary("open_long_secondary", "success")
                    log_position("open", symbol, timeframe, entry_price, "Secondary Long")
            elif latest_bar['secondary_short']:
                zone_percentage = calculate_zone_percentage(latest_bar['close'], latest_bar['wavy_l'], latest_bar['tunnel1'])
                if zone_percentage <= max_zone_entry:
                    position = 'Short'
                    entry_price = latest_bar['close']
                    logging.info(f"Secondary Strategy: Opened Short position at {entry_price:.5f}, Zone entry: {zone_percentage:.2%}")
                    update_trade_summary("open_short_secondary", "success")
                    log_position("open", symbol, timeframe, entry_price, "Secondary Short")
        else:
            if position == 'Long':
                if latest_bar['close'] < latest_bar['wavy_l']:
                    profit = latest_bar['close'] - entry_price
                    logging.info(f"Closed Long position. Entry: {entry_price:.5f}, Exit: {latest_bar['close']:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Long", entry_price, latest_bar['close'], profit)
                    log_position("close", symbol, timeframe, latest_bar['close'], "Long")
                    update_trade_summary("close_long", "success")
                    position = None
                elif latest_bar['close'] > latest_bar['tunnel1']:
                    profit = latest_bar['close'] - entry_price
                    logging.info(f"Take Profit: Closed Long position. Entry: {entry_price:.5f}, Exit: {latest_bar['close']:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Long TP", entry_price, latest_bar['close'], profit)
                    log_position("close", symbol, timeframe, latest_bar['close'], "Long TP")
                    update_trade_summary("close_long_tp", "success")
                    position = None
            elif position == 'Short':
                if latest_bar['close'] > latest_bar['wavy_h']:
                    profit = entry_price - latest_bar['close']
                    logging.info(f"Closed Short position. Entry: {entry_price:.5f}, Exit: {latest_bar['close']:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Short", entry_price, latest_bar['close'], profit)
                    log_position("close", symbol, timeframe, latest_bar['close'], "Short")
                    update_trade_summary("close_short", "success")
                    position = None
                elif latest_bar['close'] < latest_bar['tunnel1']:
                    profit = entry_price - latest_bar['close']
                    logging.info(f"Take Profit: Closed Short position. Entry: {entry_price:.5f}, Exit: {latest_bar['close']:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Short TP", entry_price, latest_bar['close'], profit)
                    log_position("close", symbol, timeframe, latest_bar['close'], "Short TP")
                    update_trade_summary("close_short_tp", "success")
                    position = None

        logging.info(f"Current position: {position}")
        logging.info("------------------------")

        time.sleep(timeframe)

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe

    logging.info(f"Starting Combined Wavy Tunnel Strategy on {symbol}, {timeframe} timeframe")

    initialize_metrics()

    try:
        run_combined_strategy(symbol, timeframe)
    except KeyboardInterrupt:
        logging.info("Strategy stopped by user")
    finally:
        print_performance_metrics()
        mt5.shutdown()
        logging.info("MetaTrader 5 connection closed")