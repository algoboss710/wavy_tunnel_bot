import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(filename='peak_dip_strategy_log.txt', level=logging.INFO,
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

def detect_peaks_dips(data, window_size=21):
    data['high_peak'] = data['high'].rolling(window=window_size, center=True).max()
    data['low_dip'] = data['low'].rolling(window=window_size, center=True).min()

    data['is_peak'] = (data['high'] == data['high_peak']) & (data['high'].shift(1) != data['high_peak'].shift(1))
    data['is_dip'] = (data['low'] == data['low_dip']) & (data['low'].shift(1) != data['low_dip'].shift(1))

    return data

def make_trading_decision(row, last_peak, last_dip):
    decision = 'Hold'
    if row['is_peak']:
        last_peak = row['high']
    if row['is_dip']:
        last_dip = row['low']

    if last_peak and last_dip:
        mid_point = (last_peak + last_dip) / 2
        if row['close'] > mid_point and row['close'] < last_peak:
            decision = 'Buy'
        elif row['close'] < mid_point and row['close'] > last_dip:
            decision = 'Sell'

    return decision, last_peak, last_dip

def mock_trade_execution(decision, current_position):
    if decision == 'Buy' and current_position != 'Long':
        return 'Long'
    elif decision == 'Sell' and current_position != 'Short':
        return 'Short'
    elif decision == 'Hold':
        return current_position
    else:
        return 'None'

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

    with open("peak_dip_successful_trades.log", "a") as f:
        f.write(json.dumps(trade_details) + "\n")

def log_position(action, symbol, timeframe, price, position_type):
    log_entry = {
        "time": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "timeframe": str(timeframe),
        "price": price,
        "position_type": position_type
    }

    with open("peak_dip_positions.log", "a") as f:
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

def run_peak_dip_strategy(symbol, timeframe):
    current_position = 'None'
    last_peak = None
    last_dip = None
    entry_price = None

    while True:
        data = get_historical_data(symbol, timeframe, 1000)
        data = detect_peaks_dips(data)

        latest_bar = data.iloc[-1]

        decision, last_peak, last_dip = make_trading_decision(latest_bar, last_peak, last_dip)
        new_position = mock_trade_execution(decision, current_position)

        logging.info(f"Time: {latest_bar['time']}, Close: {latest_bar['close']:.5f}, Decision: {decision}, Position: {new_position}")

        if latest_bar['is_peak']:
            logging.info(f"Peak detected at {latest_bar['high']:.5f}")
        if latest_bar['is_dip']:
            logging.info(f"Dip detected at {latest_bar['low']:.5f}")

        if new_position != current_position:
            if current_position != 'None':
                # Close the current position
                profit = latest_bar['close'] - entry_price if current_position == 'Long' else entry_price - latest_bar['close']
                logging.info(f"Closing {current_position} position. Entry: {entry_price:.5f}, Exit: {latest_bar['close']:.5f}, Profit: {profit:.5f}")
                log_successful_trade(symbol, timeframe, f"Close {current_position}", entry_price, latest_bar['close'], profit)
                log_position("close", symbol, timeframe, latest_bar['close'], current_position)
                update_trade_summary(f"close_{current_position.lower()}", "success")

            if new_position != 'None':
                # Open a new position
                entry_price = latest_bar['close']
                logging.info(f"Opening {new_position} position at {entry_price:.5f}")
                log_position("open", symbol, timeframe, entry_price, new_position)
                update_trade_summary(f"open_{new_position.lower()}", "success")

            current_position = new_position

        logging.info("------------------------")

        time.sleep(timeframe)

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe

    logging.info(f"Starting Peak and Dip Detection Strategy on {symbol}, {timeframe} timeframe")

    initialize_metrics()

    try:
        run_peak_dip_strategy(symbol, timeframe)
    except KeyboardInterrupt:
        logging.info("Strategy stopped by user")
    finally:
        print_performance_metrics()
        mt5.shutdown()
        logging.info("MetaTrader 5 connection closed")