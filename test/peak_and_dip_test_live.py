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

def run_peak_dip_strategy(symbol, timeframe):
    data = get_historical_data(symbol, timeframe, 1000)
    data = detect_peaks_dips(data)

    current_position = 'None'
    last_peak = None
    last_dip = None

    for index, row in data.iterrows():
        decision, last_peak, last_dip = make_trading_decision(row, last_peak, last_dip)
        current_position = mock_trade_execution(decision, current_position)

        print(f"Time: {row['time']}, Close: {row['close']}, Decision: {decision}, Position: {current_position}")

        if row['is_peak']:
            print(f"Peak detected at {row['high']}")
        if row['is_dip']:
            print(f"Dip detected at {row['low']}")

    return data

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe

    print(f"Running Peak and Dip Detection Strategy on {symbol}, {timeframe} timeframe")
    result_data = run_peak_dip_strategy(symbol, timeframe)

    print("\nStrategy test completed.")
    mt5.shutdown()