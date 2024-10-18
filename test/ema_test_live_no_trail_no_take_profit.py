import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
import os
import concurrent.futures

# Create test_logs folder if it doesn't exist
os.makedirs('test_logs_ntp_nt', exist_ok=True)

# Set up logging for successful trades
success_logger = logging.getLogger('successful_trades')
success_logger.setLevel(logging.INFO)
success_handler = logging.FileHandler('test_logs_ntp_nt/successful_trades.log')
success_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
success_logger.addHandler(success_handler)

# Set up logging for failed trades
failure_logger = logging.getLogger('failed_trades')
failure_logger.setLevel(logging.ERROR)
failure_handler = logging.FileHandler('test_logs_ntp_nt/failed_trades.log')
failure_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
failure_logger.addHandler(failure_handler)

# Global variables to track trades and performance
trade_summary = defaultdict(int)
initial_balance = 0

# Connect to MetaTrader 5
if not mt5.initialize():
    logging.error("MetaTrader5 initialization failed")
    mt5.shutdown()
    quit()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_historical_data(symbol, timeframe, num_bars):
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if bars is None or len(bars) < num_bars:
        logging.warning(f"Failed to retrieve {num_bars} bars for {symbol} on {timeframe} timeframe")
        return None
    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    logging.info(f"Retrieved {len(df)} bars for {symbol} on {timeframe} timeframe")
    return df

def wavy_tunnel_strategy(symbol, timeframe):
    data = get_historical_data(symbol, timeframe, 200)
    if data is None or len(data) < 200:
        logging.warning("Not enough data for strategy calculation")
        return None

    data['wavy_h'] = calculate_ema(data['high'], 34)
    data['wavy_c'] = calculate_ema(data['close'], 34)
    data['wavy_l'] = calculate_ema(data['low'], 34)
    data['tunnel1'] = calculate_ema(data['close'], 144)
    data['tunnel2'] = calculate_ema(data['close'], 169)
    data['rsi'] = calculate_rsi(data)

    max_wavy = data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)
    min_wavy = data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)
    max_tunnel = data[['tunnel1', 'tunnel2']].max(axis=1)
    min_tunnel = data[['tunnel1', 'tunnel2']].min(axis=1)

    rsi_upper = 70
    rsi_lower = 30

    data['long_condition'] = (data['open'] > max_wavy) & (min_wavy > max_tunnel) & (data['rsi'] < rsi_upper)
    data['short_condition'] = (data['open'] < min_wavy) & (max_wavy < min_tunnel) & (data['rsi'] > rsi_lower)
    data['exit_long'] = data['close'] < min_wavy
    data['exit_short'] = data['close'] > max_wavy

    return data

def open_position(symbol, order_type, lot_size, tp_distance, logger):
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    point = mt5.symbol_info(symbol).point

    tp_price = price + tp_distance * point if order_type == mt5.ORDER_TYPE_BUY else price - tp_distance * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "tp": tp_price,
        "deviation": 20,
        "magic": 234000,
        "comment": "Wavy Tunnel Strategy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        failure_logger.error(f"Order opening failed for {symbol}: {result.comment}")
        return False
    logger.info(f"Order opened for {symbol}: {result.order}, Price: {price}, Volume: {lot_size}, TP: {tp_price}")

    return True

def close_position(position, logger):
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
        failure_logger.error(f"Order closing failed for {position.symbol}: {result.comment}")
        return False
    logger.info(f"Order closed for {position.symbol}: {result.order}, Price: {request['price']}, Volume: {position.volume}, Profit: {position.profit}")
    return True

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
    success_logger.info(json.dumps(trade_details))

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
    logging.info("Trade Summary:")
    for action, count in trade_summary.items():
        logging.info(f"  {action}: {count}")

def log_position(position, action, symbol, timeframe, logger):
    log_entry = {
        "time": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "timeframe": str(timeframe),
        "ticket": position.ticket,
        "type": "Long" if position.type == mt5.POSITION_TYPE_BUY else "Short",
        "volume": position.volume,
        "open_price": position.price_open,
        "current_price": position.price_current,
        "profit": position.profit
    }

    if action == "close":
        log_entry["close_price"] = position.price_current

    logger.info(json.dumps(log_entry))

def trade(symbol, timeframe, lot_size=0.01, tp_distance=100):
    logger = logging.getLogger(f'{symbol}_{timeframe}')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'test_logs_ntp_nt/{symbol}_{timeframe}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    data = wavy_tunnel_strategy(symbol, timeframe)
    if data is None:
        logger.warning("Strategy calculation failed, skipping trade evaluation")
        return

    last_row = data.iloc[-1]
    logger.info(f"Trade evaluation for {symbol} on {timeframe} timeframe:")

    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.error("Failed to retrieve positions")
        return

    if positions:
        for position in positions:
            position_type = 'Long' if position.type == mt5.POSITION_TYPE_BUY else 'Short'
            logger.info(f"Current position: Type: {position_type}, Volume: {position.volume}, Open Price: {position.price_open}, Current Price: {position.price_current}, Profit: {position.profit}")

            if (position.type == mt5.POSITION_TYPE_BUY and last_row['exit_long']) or \
               (position.type == mt5.POSITION_TYPE_SELL and last_row['exit_short']):
                logger.info(f"Attempting to close {symbol} position on {timeframe} timeframe")
                logger.info(f"Exit reason: {'Price crossed below wavy tunnel' if position.type == mt5.POSITION_TYPE_BUY else 'Price crossed above wavy tunnel'}")
                update_trade_summary("close", "attempt")
                if close_position(position, logger):
                    logger.info(f"Successfully closed {symbol} position on {timeframe} timeframe")
                    update_trade_summary("close", "success")
                    log_successful_trade(symbol, timeframe, f"Close {position_type}", position.price_open, position.price_current, position.profit)
                    log_position(position, "close", symbol, timeframe, logger)
                else:
                    failure_logger.error(f"Failed to close {symbol} position on {timeframe} timeframe")
                    update_trade_summary("close", "failure")
            else:
                logger.info(f"Holding current position, exit conditions not met")
    else:
        logger.info("No open positions")
        if last_row['long_condition']:
            logger.info(f"Attempting to open long position for {symbol} on {timeframe} timeframe")
            logger.info(f"Entry reason: Price above wavy tunnel, wavy tunnel above main tunnel, RSI below overbought level")
            update_trade_summary("open_long", "attempt")
            if open_position(symbol, mt5.ORDER_TYPE_BUY, lot_size, tp_distance, logger):
                logger.info(f"Successfully opened long position for {symbol} on {timeframe} timeframe")
                update_trade_summary("open_long", "success")
                log_successful_trade(symbol, timeframe, "Open Long", last_row['open'], None, None)
                new_position = mt5.positions_get(symbol=symbol)[-1]
                log_position(new_position, "open", symbol, timeframe, logger)
            else:
                failure_logger.error(f"Failed to open long position for {symbol} on {timeframe} timeframe")
                update_trade_summary("open_long", "failure")
        elif last_row['short_condition']:
            logger.info(f"Attempting to open short position for {symbol} on {timeframe} timeframe")
            logger.info(f"Entry reason: Price below wavy tunnel, wavy tunnel below main tunnel, RSI above oversold level")
            update_trade_summary("open_short", "attempt")
            if open_position(symbol, mt5.ORDER_TYPE_SELL, lot_size, tp_distance, logger):
                logger.info(f"Successfully opened short position for {symbol} on {timeframe} timeframe")
                update_trade_summary("open_short", "success")
                log_successful_trade(symbol, timeframe, "Open Short", last_row['open'], None, None)
                new_position = mt5.positions_get(symbol=symbol)[-1]
                log_position(new_position, "open", symbol, timeframe, logger)
            else:
                failure_logger.error(f"Failed to open short position for {symbol} on {timeframe} timeframe")
                update_trade_summary("open_short", "failure")
        else:
            logger.info("No trade signal, no action taken")
            if last_row['rsi'] >= 70:
                logger.info("Reason: RSI overbought")
            elif last_row['rsi'] <= 30:
                logger.info("Reason: RSI oversold")
            else:
                logger.info("Reason: Price not beyond wavy tunnel or wavy tunnel not beyond main tunnel")

    logger.info("Trade evaluation complete")
    logger.info("------------------------")

if __name__ == "__main__":
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    timeframes = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]

    logging.info("Starting Wavy Tunnel Strategy Test")
    logging.info(f"Testing on symbols: {symbols}")
    logging.info(f"Timeframes: {timeframes}")

    initialize_metrics()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(symbols) * len(timeframes)) as executor:
            while True:
                futures = []
                for symbol in symbols:
                    for tf in timeframes:
                        futures.append(executor.submit(trade, symbol, tf))

                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

                logging.info("Waiting for next tick...")
                time.sleep(1)  # Small delay to prevent excessive CPU usage
    except KeyboardInterrupt:
        logging.info("Strategy stopped by user")
    finally:
        print_performance_metrics()
        mt5.shutdown()
        logging.info("MetaTrader 5 connection closed")