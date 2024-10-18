import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(filename='liquidity_based_strategy_log.txt', level=logging.INFO,
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

def identify_liquidity_levels(data, volume_threshold=80, price_threshold=0.0005):
    data['volume_percentile'] = data['tick_volume'].rolling(window=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    high_volume_points = data[data['volume_percentile'] > volume_threshold/100]

    liquidity_levels = []
    for i, row in high_volume_points.iterrows():
        if not liquidity_levels or abs(row['close'] - liquidity_levels[-1][1]) > price_threshold:
            liquidity_levels.append((row.name, row['close']))

    logging.info(f"Identified {len(liquidity_levels)} liquidity levels")
    return liquidity_levels

def find_nearest_liquidity_level(price, levels, direction):
    if direction == 'above':
        valid_levels = [level for level in levels if level[1] > price]
    else:  # 'below'
        valid_levels = [level for level in levels if level[1] < price]

    if not valid_levels:
        return None

    return min(valid_levels, key=lambda x: abs(x[1] - price))

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

    with open("liquidity_based_successful_trades.log", "a") as f:
        f.write(json.dumps(trade_details) + "\n")

def log_position(action, symbol, timeframe, entry_price, tp_level, sl_level, position_type):
    log_entry = {
        "time": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "timeframe": str(timeframe),
        "entry_price": entry_price,
        "tp_level": tp_level,
        "sl_level": sl_level,
        "position_type": position_type
    }

    with open("liquidity_based_positions.log", "a") as f:
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

def run_liquidity_based_strategy(symbol, timeframe):
    position = None
    entry_price = None
    tp_level = None
    sl_level = None

    while True:
        data = get_historical_data(symbol, timeframe, 1000)
        data = simple_entry_signal(data)

        liquidity_levels = identify_liquidity_levels(data)

        latest_bar = data.iloc[-1]

        logging.info(f"Evaluating {symbol} on {timeframe} timeframe at {latest_bar['time']}")
        logging.info(f"Current price: Close: {latest_bar['close']:.5f}, High: {latest_bar['high']:.5f}, Low: {latest_bar['low']:.5f}")

        if position is None:
            if latest_bar['buy_signal']:
                tp_level_data = find_nearest_liquidity_level(latest_bar['close'], liquidity_levels, 'above')
                sl_level_data = find_nearest_liquidity_level(latest_bar['close'], liquidity_levels, 'below')
                if tp_level_data and sl_level_data:
                    position = 'Long'
                    entry_price = latest_bar['close']
                    tp_level = tp_level_data[1]
                    sl_level = sl_level_data[1]
                    logging.info(f"Opened Long position at {entry_price:.5f}, TP set at {tp_level:.5f}, SL set at {sl_level:.5f}")
                    update_trade_summary("open_long", "success")
                    log_position("open", symbol, timeframe, entry_price, tp_level, sl_level, "Long")
                else:
                    logging.info("Couldn't set TP or SL for Long position, trade cancelled")
            elif latest_bar['sell_signal']:
                tp_level_data = find_nearest_liquidity_level(latest_bar['close'], liquidity_levels, 'below')
                sl_level_data = find_nearest_liquidity_level(latest_bar['close'], liquidity_levels, 'above')
                if tp_level_data and sl_level_data:
                    position = 'Short'
                    entry_price = latest_bar['close']
                    tp_level = tp_level_data[1]
                    sl_level = sl_level_data[1]
                    logging.info(f"Opened Short position at {entry_price:.5f}, TP set at {tp_level:.5f}, SL set at {sl_level:.5f}")
                    update_trade_summary("open_short", "success")
                    log_position("open", symbol, timeframe, entry_price, tp_level, sl_level, "Short")
                else:
                    logging.info("Couldn't set TP or SL for Short position, trade cancelled")
        else:
            if position == 'Long':
                if latest_bar['high'] >= tp_level:
                    profit = tp_level - entry_price
                    logging.info(f"Take Profit hit for Long position. Entry: {entry_price:.5f}, Exit: {tp_level:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Long TP", entry_price, tp_level, profit)
                    update_trade_summary("close_long_tp", "success")
                    position = None
                elif latest_bar['low'] <= sl_level:
                    loss = sl_level - entry_price
                    logging.info(f"Stop Loss hit for Long position. Entry: {entry_price:.5f}, Exit: {sl_level:.5f}, Loss: {loss:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Long SL", entry_price, sl_level, loss)
                    update_trade_summary("close_long_sl", "success")
                    position = None
            elif position == 'Short':
                if latest_bar['low'] <= tp_level:
                    profit = entry_price - tp_level
                    logging.info(f"Take Profit hit for Short position. Entry: {entry_price:.5f}, Exit: {tp_level:.5f}, Profit: {profit:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Short TP", entry_price, tp_level, profit)
                    update_trade_summary("close_short_tp", "success")
                    position = None
                elif latest_bar['high'] >= sl_level:
                    loss = entry_price - sl_level
                    logging.info(f"Stop Loss hit for Short position. Entry: {entry_price:.5f}, Exit: {sl_level:.5f}, Loss: {loss:.5f}")
                    log_successful_trade(symbol, timeframe, "Close Short SL", entry_price, sl_level, loss)
                    update_trade_summary("close_short_sl", "success")
                    position = None

        logging.info(f"Current position: {position}")
        logging.info("------------------------")

        time.sleep(timeframe)

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe

    logging.info(f"Starting Liquidity-based Strategy on {symbol}, {timeframe} timeframe")

    initialize_metrics()

    try:
        run_liquidity_based_strategy(symbol, timeframe)
    except KeyboardInterrupt:
        logging.info("Strategy stopped by user")
    finally:
        print_performance_metrics()
        mt5.shutdown()
        logging.info("MetaTrader 5 connection closed")