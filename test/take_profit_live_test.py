import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(filename='sr_based_strategy_log.txt', level=logging.INFO,
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

def identify_support_resistance(data, window=14, threshold=0.01):
    data['rolling_high'] = data['high'].rolling(window=window, center=True).max()
    data['rolling_low'] = data['low'].rolling(window=window, center=True).min()

    supports = []
    resistances = []

    for i in range(window, len(data) - window):
        if data['low'].iloc[i] == data['rolling_low'].iloc[i]:
            supports.append((data.index[i], data['low'].iloc[i]))
        if data['high'].iloc[i] == data['rolling_high'].iloc[i]:
            resistances.append((data.index[i], data['high'].iloc[i]))

    # Filter out levels that are too close to each other
    filtered_supports = [supports[0]]
    filtered_resistances = [resistances[0]]

    for s in supports[1:]:
        if abs(s[1] - filtered_supports[-1][1]) / filtered_supports[-1][1] > threshold:
            filtered_supports.append(s)

    for r in resistances[1:]:
        if abs(r[1] - filtered_resistances[-1][1]) / filtered_resistances[-1][1] > threshold:
            filtered_resistances.append(r)

    logging.info(f"Identified {len(filtered_supports)} support levels and {len(filtered_resistances)} resistance levels")
    return filtered_supports, filtered_resistances

def find_nearest_level(price, levels):
    return min(levels, key=lambda x: abs(x[1] - price))

def simple_entry_signal(data):
    data['sma_short'] = data['close'].rolling(window=10).mean()
    data['sma_long'] = data['close'].rolling(window=30).mean()
    data['buy_signal'] = (data['sma_short'] > data['sma_long']) & (data['sma_short'].shift(1) <= data['sma_long'].shift(1))
    data['sell_signal'] = (data['sma_short'] < data['sma_long']) & (data['sma_short'].shift(1) >= data['sma_long'].shift(1))
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

    with open("sr_based_successful_trades.log", "a") as f:
        f.write(json.dumps(trade_details) + "\n")

def log_position(action, symbol, timeframe, entry_price, tp_level):
    log_entry = {
        "time": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "timeframe": str(timeframe),
        "entry_price": entry_price,
        "tp_level": tp_level
    }

    with open("sr_based_positions.log", "a") as f:
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

def run_sr_based_tp_strategy(symbol, timeframe):
    position = None
    entry_price = None
    tp_level = None
    supports = []
    resistances = []

    while True:
        # Get the latest data
        data = get_historical_data(symbol, timeframe, 1000)
        data = simple_entry_signal(data)

        # Update support and resistance levels periodically
        if len(supports) == 0 or len(resistances) == 0 or len(data) % 100 == 0:
            supports, resistances = identify_support_resistance(data)

        # Get the latest bar
        latest_bar = data.iloc[-1]

        logging.info(f"Evaluating {symbol} on {timeframe} timeframe at {latest_bar['time']}")
        logging.info(f"Current price: Close: {latest_bar['close']:.5f}, High: {latest_bar['high']:.5f}, Low: {latest_bar['low']:.5f}")

        if position is None:
            if latest_bar['buy_signal']:
                position = 'Long'
                entry_price = latest_bar['close']
                _, tp_level = find_nearest_level(entry_price, [r for r in resistances if r[1] > entry_price])
                logging.info(f"Opening Long position at {entry_price:.5f}, TP set at {tp_level:.5f}")
                update_trade_summary("open_long", "success")
                log_position("open", symbol, timeframe, entry_price, tp_level)
            elif latest_bar['sell_signal']:
                position = 'Short'
                entry_price = latest_bar['close']
                _, tp_level = find_nearest_level(entry_price, [s for s in supports if s[1] < entry_price])
                logging.info(f"Opening Short position at {entry_price:.5f}, TP set at {tp_level:.5f}")
                update_trade_summary("open_short", "success")
                log_position("open", symbol, timeframe, entry_price, tp_level)
        else:
            if position == 'Long':
                if latest_bar['high'] >= tp_level:
                    profit = tp_level - entry_price
                    logging.info(f"Take Profit hit for Long position. Entry: {entry_price:.5f}, Exit: {tp_level:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Long", entry_price, tp_level, profit)
                    log_position("close", symbol, timeframe, entry_price, tp_level)
                    update_trade_summary("close_long", "success")
                    position = None
                    entry_price = None
                    tp_level = None
            elif position == 'Short':
                if latest_bar['low'] <= tp_level:
                    profit = entry_price - tp_level
                    logging.info(f"Take Profit hit for Short position. Entry: {entry_price:.5f}, Exit: {tp_level:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Short", entry_price, tp_level, profit)
                    log_position("close", symbol, timeframe, entry_price, tp_level)
                    update_trade_summary("close_short", "success")
                    position = None
                    entry_price = None
                    tp_level = None

        logging.info(f"Current position: {position}")
        logging.info("------------------------")

        # Wait for the next bar
        time.sleep(timeframe)

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe

    logging.info(f"Starting Support/Resistance-based Take-Profit Strategy on {symbol}, {timeframe} timeframe")

    initialize_metrics()

    try:
        run_sr_based_tp_strategy(symbol, timeframe)
    except KeyboardInterrupt:
        logging.info("Strategy stopped by user")
    finally:
        print_performance_metrics()
        mt5.shutdown()
        logging.info("MetaTrader 5 connection closed")