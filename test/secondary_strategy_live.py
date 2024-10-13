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

def run_combined_strategy(symbol, timeframe, max_zone_entry=0.25):
    data = get_historical_data(symbol, timeframe, 1000)
    data = wavy_tunnel_strategy(data)

    position = None
    entry_price = None

    for index, row in data.iterrows():
        if position is None:
            if row['primary_long']:
                position = 'Long'
                entry_price = row['close']
                print(f"Primary Strategy: Opened Long position at {entry_price:.5f}")
            elif row['primary_short']:
                position = 'Short'
                entry_price = row['close']
                print(f"Primary Strategy: Opened Short position at {entry_price:.5f}")
            elif row['secondary_long']:
                zone_percentage = calculate_zone_percentage(row['close'], row['wavy_h'], row['tunnel1'])
                if zone_percentage <= max_zone_entry:
                    position = 'Long'
                    entry_price = row['close']
                    print(f"Secondary Strategy: Opened Long position at {entry_price:.5f}, Zone entry: {zone_percentage:.2%}")
            elif row['secondary_short']:
                zone_percentage = calculate_zone_percentage(row['close'], row['wavy_l'], row['tunnel1'])
                if zone_percentage <= max_zone_entry:
                    position = 'Short'
                    entry_price = row['close']
                    print(f"Secondary Strategy: Opened Short position at {entry_price:.5f}, Zone entry: {zone_percentage:.2%}")
        else:
            if position == 'Long':
                if row['close'] < row['wavy_l']:
                    print(f"Closed Long position. Entry: {entry_price:.5f}, Exit: {row['close']:.5f}")
                    position = None
                elif row['close'] > row['tunnel1']:
                    print(f"Take Profit: Closed Long position. Entry: {entry_price:.5f}, Exit: {row['close']:.5f}")
                    position = None
            elif position == 'Short':
                if row['close'] > row['wavy_h']:
                    print(f"Closed Short position. Entry: {entry_price:.5f}, Exit: {row['close']:.5f}")
                    position = None
                elif row['close'] < row['tunnel1']:
                    print(f"Take Profit: Closed Short position. Entry: {entry_price:.5f}, Exit: {row['close']:.5f}")
                    position = None

        print(f"Time: {row['time']}, Close: {row['close']:.5f}, Position: {position}")

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe

    print(f"Running Combined Wavy Tunnel Strategy on {symbol}, {timeframe} timeframe")
    run_combined_strategy(symbol, timeframe)

    print("\nStrategy test completed.")
    mt5.shutdown()