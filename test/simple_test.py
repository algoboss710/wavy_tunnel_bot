import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta

# Connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
    quit()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def get_historical_data(symbol, timeframe, num_bars):
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def wavy_tunnel_strategy(symbol, timeframe):
    data = get_historical_data(symbol, timeframe, 200)

    data['wavy_h'] = calculate_ema(data['high'], 34)
    data['wavy_c'] = calculate_ema(data['close'], 34)
    data['wavy_l'] = calculate_ema(data['low'], 34)
    data['tunnel1'] = calculate_ema(data['close'], 144)
    data['tunnel2'] = calculate_ema(data['close'], 169)

    data['long_condition'] = (data['open'] > data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)) & \
                             (data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1) > data[['tunnel1', 'tunnel2']].max(axis=1))

    data['short_condition'] = (data['open'] < data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)) & \
                              (data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1) < data[['tunnel1', 'tunnel2']].min(axis=1))

    data['exit_long'] = data['close'] < data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)
    data['exit_short'] = data['close'] > data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)

    return data.iloc[-1]

def open_position(symbol, order_type, lot_size):
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "Wavy Tunnel Strategy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order opening failed: {result.comment}")
        return False
    print(f"Order opened: {result.order}")
    return True

def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": tick.bid if position.type == 0 else tick.ask,
        "deviation": 20,
        "magic": 234000,
        "comment": "Wavy Tunnel Strategy - Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order closing failed: {result.comment}")
        return False
    print(f"Order closed: {result.order}")
    return True

def trade(symbol, timeframe, lot_size=0.01):
    last_row = wavy_tunnel_strategy(symbol, timeframe)

    print(f"Evaluating {symbol} on {timeframe} timeframe")
    print(f"Long condition: {last_row['long_condition']}")
    print(f"Short condition: {last_row['short_condition']}")
    print(f"Exit long condition: {last_row['exit_long']}")
    print(f"Exit short condition: {last_row['exit_short']}")

    positions = mt5.positions_get(symbol=symbol)

    if positions:
        for position in positions:
            if (position.type == mt5.POSITION_TYPE_BUY and last_row['exit_long']) or \
               (position.type == mt5.POSITION_TYPE_SELL and last_row['exit_short']):
                print(f"Closing {symbol} position on {timeframe} timeframe")
                close_position(position)
    else:
        if last_row['long_condition']:
            print(f"Opening long position for {symbol} on {timeframe} timeframe")
            open_position(symbol, mt5.ORDER_TYPE_BUY, lot_size)
        elif last_row['short_condition']:
            print(f"Opening short position for {symbol} on {timeframe} timeframe")
            open_position(symbol, mt5.ORDER_TYPE_SELL, lot_size)

if __name__ == "__main__":
    symbol = "EURUSD"
    timeframes = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]

    print("Starting Wavy Tunnel Strategy Test")
    print(f"Testing on symbol: {symbol}")
    print(f"Timeframes: {timeframes}")
    print("Press Ctrl+C to stop the script")

    try:
        while True:
            for tf in timeframes:
                trade(symbol, tf)
            print("\nWaiting for next tick...\n")
            time.sleep(1)  # Small delay to prevent excessive CPU usage
    except KeyboardInterrupt:
        print("\nStrategy stopped by user")
    finally:
        mt5.shutdown()
        print("MetaTrader 5 connection closed")