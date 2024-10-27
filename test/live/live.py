import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import time

# Setup logging
os.makedirs('live_trading_logs', exist_ok=True)
logging.basicConfig(filename='live_trading_logs/live_trading.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

def initialize_mt5():
    if not mt5.initialize():
        logging.error("MetaTrader5 initialization failed")
        return False
    logging.info("MetaTrader5 initialized successfully")
    return True

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period):
    tr = pd.concat([high - low,
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def get_latest_data(symbol, timeframe, num_bars=200):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        logging.warning(f"Failed to retrieve data for {symbol} on {timeframe} timeframe")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def wavy_tunnel_strategy(df, params):
    df['wavy_h'] = calculate_ema(df['high'], params['wavy_ema'])
    df['wavy_c'] = calculate_ema(df['close'], params['wavy_ema'])
    df['wavy_l'] = calculate_ema(df['low'], params['wavy_ema'])
    df['tunnel1'] = calculate_ema(df['close'], params['tunnel_ema1'])
    df['tunnel2'] = calculate_ema(df['close'], params['tunnel_ema2'])

    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], params['atr_period'])
    df['threshold'] = df['atr'] * params['atr_multiplier']

    max_wavy = df[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)
    min_wavy = df[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)
    max_tunnel = df[['tunnel1', 'tunnel2']].max(axis=1)
    min_tunnel = df[['tunnel1', 'tunnel2']].min(axis=1)

    df['long_condition'] = (df['open'] > max_wavy + df['threshold']) & (min_wavy > max_tunnel)
    df['short_condition'] = (df['open'] < min_wavy - df['threshold']) & (max_wavy < min_tunnel)

    if params['apply_rsi_filter']:
        df['rsi'] = calculate_rsi(df, params['rsi_period'])
        df['long_condition'] &= df['rsi'] < params['rsi_upper']
        df['short_condition'] &= df['rsi'] > params['rsi_lower']

    df['exit_long'] = df['close'] < min_wavy
    df['exit_short'] = df['close'] > max_wavy

    return df

def place_market_order(symbol, order_type, volume, stop_loss=None, take_profit=None):
    order_type_map = {
        "buy": mt5.ORDER_TYPE_BUY,
        "sell": mt5.ORDER_TYPE_SELL
    }
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type_map[order_type],
        "price": mt5.symbol_info_tick(symbol).ask if order_type == "buy" else mt5.symbol_info_tick(symbol).bid,
        "deviation": 20,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if stop_loss:
        request["sl"] = stop_loss
    if take_profit:
        request["tp"] = take_profit

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order placement failed: {result.comment}")
        return None
    return result

def close_position(position):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
        "deviation": 20,
        "magic": 234000,
        "comment": "python script close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Position closure failed: {result.comment}")
        return False
    return True

def run_live_trading(symbol_config):
    symbol = symbol_config['currency_pair']
    timeframe = getattr(mt5, f"TIMEFRAME_{symbol_config['timeframe']}")

    while True:
        try:
            df = get_latest_data(symbol, timeframe)
            if df is None:
                logging.warning(f"No data for {symbol} on {timeframe} timeframe, retrying in 60 seconds")
                time.sleep(60)
                continue

            df = wavy_tunnel_strategy(df, symbol_config)

            # Check for open positions
            positions = mt5.positions_get(symbol=symbol)

            # Close positions if exit conditions are met
            for position in positions:
                if (position.type == 0 and df['exit_long'].iloc[-1]) or \
                   (position.type == 1 and df['exit_short'].iloc[-1]):
                    if close_position(position):
                        logging.info(f"Closed {symbol} position at {position.price_current}")

            # Open new positions if entry conditions are met
            if df['long_condition'].iloc[-1] and not any(p.type == 0 for p in positions):
                result = place_market_order(symbol, "buy", symbol_config['lot_size'])
                if result:
                    logging.info(f"Opened long position for {symbol} at {result.price}")
            elif df['short_condition'].iloc[-1] and not any(p.type == 1 for p in positions):
                result = place_market_order(symbol, "sell", symbol_config['lot_size'])
                if result:
                    logging.info(f"Opened short position for {symbol} at {result.price}")

            # Wait for the next candle
            time.sleep(timeframe)

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    if not initialize_mt5():
        exit(1)

    try:
        for symbol_config in config['symbols']:
            run_live_trading(symbol_config)
    except KeyboardInterrupt:
        logging.info("Live trading interrupted by user")
    finally:
        mt5.shutdown()
        logging.info("MetaTrader 5 connection closed")