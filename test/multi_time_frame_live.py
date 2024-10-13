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

def run_multi_timeframe_strategy(symbol, timeframes):
    data = {}
    signals = {}

    for tf in timeframes:
        data[tf] = get_historical_data(symbol, tf, 1000)
        signals[tf] = wavy_tunnel_signals(data[tf])

    position = None
    entry_price = None

    # Use the shortest timeframe for trade execution
    shortest_tf = min(timeframes)

    for index, row in signals[shortest_tf].iterrows():
        # Check if signals align across all timeframes
        long_signals = all(signals[tf].loc[signals[tf].index <= index, 'long_condition'].iloc[-1] for tf in timeframes)
        short_signals = all(signals[tf].loc[signals[tf].index <= index, 'short_condition'].iloc[-1] for tf in timeframes)

        if position is None:
            if long_signals:
                position = 'Long'
                entry_price = row['close']
                print(f"Opened Long position at {entry_price:.5f}")
            elif short_signals:
                position = 'Short'
                entry_price = row['close']
                print(f"Opened Short position at {entry_price:.5f}")
        else:
            if position == 'Long':
                if row['close'] < row['wavy_l']:
                    print(f"Closed Long position. Entry: {entry_price:.5f}, Exit: {row['close']:.5f}")
                    position = None
            elif position == 'Short':
                if row['close'] > row['wavy_h']:
                    print(f"Closed Short position. Entry: {entry_price:.5f}, Exit: {row['close']:.5f}")
                    position = None

        print(f"Time: {row['time']}, Close: {row['close']:.5f}, Position: {position}")

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]

    print(f"Running Multi-Timeframe Wavy Tunnel Strategy on {symbol}")
    print(f"Timeframes: {timeframes}")
    run_multi_timeframe_strategy(symbol, timeframes)

    print("\nStrategy test completed.")
    mt5.shutdown()