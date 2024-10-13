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

    return filtered_supports, filtered_resistances

def find_nearest_level(price, levels):
    return min(levels, key=lambda x: abs(x[1] - price))

def simple_entry_signal(data):
    # Simple moving average crossover for demonstration
    data['sma_short'] = data['close'].rolling(window=10).mean()
    data['sma_long'] = data['close'].rolling(window=30).mean()
    data['buy_signal'] = (data['sma_short'] > data['sma_long']) & (data['sma_short'].shift(1) <= data['sma_long'].shift(1))
    data['sell_signal'] = (data['sma_short'] < data['sma_long']) & (data['sma_short'].shift(1) >= data['sma_long'].shift(1))
    return data

def run_sr_based_tp_strategy(symbol, timeframe):
    data = get_historical_data(symbol, timeframe, 1000)
    data = simple_entry_signal(data)

    supports, resistances = identify_support_resistance(data)

    position = None
    entry_price = None
    tp_level = None

    for index, row in data.iterrows():
        if position is None:
            if row['buy_signal']:
                position = 'Long'
                entry_price = row['close']
                _, tp_level = find_nearest_level(entry_price, [r for r in resistances if r[1] > entry_price])
                print(f"Opened Long position at {entry_price:.5f}, TP set at {tp_level:.5f}")
            elif row['sell_signal']:
                position = 'Short'
                entry_price = row['close']
                _, tp_level = find_nearest_level(entry_price, [s for s in supports if s[1] < entry_price])
                print(f"Opened Short position at {entry_price:.5f}, TP set at {tp_level:.5f}")
        else:
            if position == 'Long':
                if row['high'] >= tp_level:
                    print(f"Take Profit hit for Long position. Entry: {entry_price:.5f}, Exit: {tp_level:.5f}")
                    position = None
                    entry_price = None
                    tp_level = None
            elif position == 'Short':
                if row['low'] <= tp_level:
                    print(f"Take Profit hit for Short position. Entry: {entry_price:.5f}, Exit: {tp_level:.5f}")
                    position = None
                    entry_price = None
                    tp_level = None

        print(f"Time: {row['time']}, Close: {row['close']:.5f}, Position: {position}")

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe

    print(f"Running Support/Resistance-based Take-Profit Strategy on {symbol}, {timeframe} timeframe")
    run_sr_based_tp_strategy(symbol, timeframe)

    print("\nStrategy test completed.")
    mt5.shutdown()