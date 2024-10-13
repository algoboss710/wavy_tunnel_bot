import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Initialize connection to MetaTrader 5
if not mt5.initialize():
    print("MetaTrader 5 initialization failed")
    quit()

def get_historical_data(symbol, timeframe, num_bars):
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def identify_liquidity_levels(data, volume_threshold=80, price_threshold=0.0005):
    data['volume_percentile'] = data['tick_volume'].rolling(window=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    high_volume_points = data[data['volume_percentile'] > volume_threshold/100]

    liquidity_levels = []
    for i, row in high_volume_points.iterrows():
        if not liquidity_levels or abs(row['close'] - liquidity_levels[-1][1]) > price_threshold:
            liquidity_levels.append((row.name, row['close']))

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

def run_liquidity_based_strategy(symbol, timeframe):
    data = get_historical_data(symbol, timeframe, 1000)
    data = simple_entry_signal(data)

    liquidity_levels = identify_liquidity_levels(data)

    position = None
    entry_price = None
    tp_level = None
    sl_level = None

    for index, row in data.iterrows():
        if position is None:
            if row['buy_signal']:
                position = 'Long'
                entry_price = row['close']
                tp_level_data = find_nearest_liquidity_level(entry_price, liquidity_levels, 'above')
                sl_level_data = find_nearest_liquidity_level(entry_price, liquidity_levels, 'below')
                if tp_level_data and sl_level_data:
                    tp_level = tp_level_data[1]
                    sl_level = sl_level_data[1]
                    print(f"Opened Long position at {entry_price:.5f}, TP set at {tp_level:.5f}, SL set at {sl_level:.5f}")
                else:
                    position = None  # Cancel the trade if we can't set TP or SL
            elif row['sell_signal']:
                position = 'Short'
                entry_price = row['close']
                tp_level_data = find_nearest_liquidity_level(entry_price, liquidity_levels, 'below')
                sl_level_data = find_nearest_liquidity_level(entry_price, liquidity_levels, 'above')
                if tp_level_data and sl_level_data:
                    tp_level = tp_level_data[1]
                    sl_level = sl_level_data[1]
                    print(f"Opened Short position at {entry_price:.5f}, TP set at {tp_level:.5f}, SL set at {sl_level:.5f}")
                else:
                    position = None  # Cancel the trade if we can't set TP or SL
        else:
            if position == 'Long':
                if row['high'] >= tp_level:
                    print(f"Take Profit hit for Long position. Entry: {entry_price:.5f}, Exit: {tp_level:.5f}")
                    position = None
                elif row['low'] <= sl_level:
                    print(f"Stop Loss hit for Long position. Entry: {entry_price:.5f}, Exit: {sl_level:.5f}")
                    position = None
            elif position == 'Short':
                if row['low'] <= tp_level:
                    print(f"Take Profit hit for Short position. Entry: {entry_price:.5f}, Exit: {tp_level:.5f}")
                    position = None
                elif row['high'] >= sl_level:
                    print(f"Stop Loss hit for Short position. Entry: {entry_price:.5f}, Exit: {sl_level:.5f}")
                    position = None

        print(f"Time: {row['time']}, Close: {row['close']:.5f}, Position: {position}")

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe

    print(f"Running Liquidity-based Strategy on {symbol}, {timeframe} timeframe")
    run_liquidity_based_strategy(symbol, timeframe)

    print("\nStrategy test completed.")
    mt5.shutdown()