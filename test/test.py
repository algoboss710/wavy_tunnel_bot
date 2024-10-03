import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback
import concurrent.futures
import threading
import queue
import os

# Create a logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up main logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
main_log_file = 'logs/main_strategy.log'
main_log_handler = RotatingFileHandler(main_log_file, maxBytes=5*1024*1024, backupCount=2)
main_log_handler.setFormatter(log_formatter)
main_logger = logging.getLogger('main')
main_logger.addHandler(main_log_handler)
main_logger.setLevel(logging.DEBUG)

# Set up trade logging
trade_formatter = logging.Formatter('%(asctime)s - %(message)s')
trade_file = 'logs/successful_trades.log'
trade_handler = RotatingFileHandler(trade_file, maxBytes=5*1024*1024, backupCount=2)
trade_handler.setFormatter(trade_formatter)
trade_logger = logging.getLogger('trade_logger')
trade_logger.addHandler(trade_handler)
trade_logger.setLevel(logging.INFO)

# Global lock for thread-safe operations
global_lock = threading.Lock()

# Semaphore to limit concurrent API calls
api_semaphore = threading.Semaphore(5)  # Adjust the value based on API limits

# Dictionary to store loggers for each symbol-timeframe combination
symbol_timeframe_loggers = {}

def get_logger(symbol, timeframe):
    key = f"{symbol}_{timeframe}"
    if key not in symbol_timeframe_loggers:
        logger = logging.getLogger(key)
        logger.setLevel(logging.DEBUG)
        log_file = f'logs/{key}.log'
        handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)
        symbol_timeframe_loggers[key] = logger
    return symbol_timeframe_loggers[key]

def log_trade(symbol, timeframe, message):
    trade_logger.info(f"{symbol} {timeframe}: {message}")
    success_file = f'logs/{symbol}_{timeframe}_success.log'
    with open(success_file, 'a') as f:
        f.write(f"{datetime.now()} - {message}\n")

main_logger.info("Script started. Initializing MetaTrader 5 connection...")

# Connect to MetaTrader 5
if not mt5.initialize():
    main_logger.error("Failed to initialize MetaTrader 5")
    mt5.shutdown()
else:
    main_logger.info("MetaTrader 5 connection initialized successfully")

# Strategy Parameters
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
TIMEFRAMES = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30]
LOT_SIZE = 0.01
ACCOUNT_RISK = 0.01

# EMA Periods
EMA_WAVY = 34
EMA_OUTER1 = 144
EMA_OUTER2 = 169
EMA_ADDITIONAL1 = 12
EMA_ADDITIONAL2 = 200

# Other Parameters
RSI_PERIOD = 14
ATR_PERIOD = 14
THRESHOLD_PIPS = 20
MIN_GAP_SECOND_STRATEGY = 150
MAX_ZONE_PERCENTAGE = 0.25

# Global variables for tracking
trades_data = {symbol: {tf: {"executed": 0, "analyzed": 0, "won": 0, "lost": 0, "profit": 0.0} for tf in TIMEFRAMES} for symbol in SYMBOLS}
initial_balance = 0.0

main_logger.info(f"Strategy parameters set for symbols: {SYMBOLS}")
main_logger.info(f"Timeframes: {TIMEFRAMES}")
main_logger.info(f"LOT_SIZE={LOT_SIZE}")
main_logger.info(f"EMA periods: WAVY={EMA_WAVY}, OUTER1={EMA_OUTER1}, OUTER2={EMA_OUTER2}")
main_logger.info(f"Other parameters: RSI_PERIOD={RSI_PERIOD}, ATR_PERIOD={ATR_PERIOD}, THRESHOLD_PIPS={THRESHOLD_PIPS}")

def get_account_info():
    with api_semaphore:
        account_info = mt5.account_info()
    if account_info is None:
        main_logger.error("Failed to get account info")
        return None
    return {
        "balance": account_info.balance,
        "equity": account_info.equity,
        "profit": account_info.profit,
        "margin": account_info.margin,
        "free_margin": account_info.margin_free
    }

def log_account_info(info, is_initial=False):
    if info:
        main_logger.info(f"{'Initial' if is_initial else 'Current'} Account Info:")
        main_logger.info(f"Balance: {info['balance']:.2f}")
        main_logger.info(f"Equity: {info['equity']:.2f}")
        main_logger.info(f"Profit: {info['profit']:.2f}")
        main_logger.info(f"Margin: {info['margin']:.2f}")
        main_logger.info(f"Free Margin: {info['free_margin']:.2f}")
    else:
        main_logger.warning("Unable to log account info: No data available")

def log_trade_summary(symbol, timeframe):
    logger = get_logger(symbol, timeframe)
    global trades_data
    with global_lock:
        logger.info(f"Trade Summary for {symbol} on {timeframe}:")
        logger.info(f"Trades Analyzed: {trades_data[symbol][timeframe]['analyzed']}")
        logger.info(f"Trades Executed: {trades_data[symbol][timeframe]['executed']}")
        logger.info(f"Trades Won: {trades_data[symbol][timeframe]['won']}")
        logger.info(f"Trades Lost: {trades_data[symbol][timeframe]['lost']}")
        logger.info(f"Total Profit: {trades_data[symbol][timeframe]['profit']:.2f}")

def get_data(symbol, timeframe, num_bars):
    logger = get_logger(symbol, timeframe)
    logger.info(f"Fetching {num_bars} bars of {symbol} data for {timeframe} timeframe")
    try:
        with api_semaphore:
            bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        df = pd.DataFrame(bars)[['time', 'open', 'high', 'low', 'close']]
        df['time'] = pd.to_datetime(df['time'], unit='s')
        logger.info(f"Successfully fetched {len(df)} bars of data for {symbol} on {timeframe}")
        logger.debug(f"{symbol} {timeframe} data range: from {df['time'].min()} to {df['time'].max()}")
        return df
    except Exception as e:
        logger.error(f"Error in get_data for {symbol} on {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def calculate_indicators(df, symbol, timeframe):
    logger = get_logger(symbol, timeframe)
    logger.info(f"Calculating indicators for {symbol} on {timeframe}")
    try:
        # Calculate EMAs
        df['ema_high'] = df['high'].ewm(span=EMA_WAVY, adjust=False).mean()
        df['ema_close'] = df['close'].ewm(span=EMA_WAVY, adjust=False).mean()
        df['ema_low'] = df['low'].ewm(span=EMA_WAVY, adjust=False).mean()
        df['ema_outer1'] = df['close'].ewm(span=EMA_OUTER1, adjust=False).mean()
        df['ema_outer2'] = df['close'].ewm(span=EMA_OUTER2, adjust=False).mean()
        df['ema_12'] = df['close'].ewm(span=EMA_ADDITIONAL1, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=EMA_ADDITIONAL2, adjust=False).mean()

        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Calculate ATR
        df['atr'] = df['high'] - df['low']
        df['atr'] = df['atr'].rolling(window=ATR_PERIOD).mean()

        logger.info(f"Indicators calculated successfully for {symbol} on {timeframe}")
        return df
    except Exception as e:
        logger.error(f"Error in calculate_indicators for {symbol} on {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def check_primary_entry(df, index, symbol, timeframe):
    logger = get_logger(symbol, timeframe)
    logger.info(f"Checking primary entry conditions for {symbol} on {timeframe}")
    try:
        current = df.iloc[index]
        prev = df.iloc[index - 1]

        with api_semaphore:
            point = mt5.symbol_info(symbol).point

        long_conditions = [
            current['open'] > max(current['ema_high'], current['ema_close'], current['ema_low']),
            min(current['ema_high'], current['ema_close'], current['ema_low']) > max(current['ema_outer1'], current['ema_outer2']),
            current['rsi'] < 70,
            current['atr'] < 2 * df['atr'].rolling(window=ATR_PERIOD).mean().iloc[index],
            current['open'] > max(current['ema_high'], current['ema_close'], current['ema_low']) + THRESHOLD_PIPS * point
        ]

        short_conditions = [
            current['open'] < min(current['ema_high'], current['ema_close'], current['ema_low']),
            max(current['ema_high'], current['ema_close'], current['ema_low']) < min(current['ema_outer1'], current['ema_outer2']),
            current['rsi'] > 30,
            current['atr'] < 2 * df['atr'].rolling(window=ATR_PERIOD).mean().iloc[index],
            current['open'] < min(current['ema_high'], current['ema_close'], current['ema_low']) - THRESHOLD_PIPS * point
        ]

        long_condition = all(long_conditions)
        short_condition = all(short_conditions)

        logger.debug(f"{symbol} {timeframe} Primary entry conditions: Long={long_condition}, Short={short_condition}")

        return long_condition, short_condition
    except Exception as e:
        logger.error(f"Error in check_primary_entry for {symbol} on {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, False

def check_secondary_entry(df, index, symbol, timeframe):
    logger = get_logger(symbol, timeframe)
    logger.info(f"Checking secondary entry conditions for {symbol} on {timeframe}")
    try:
        current = df.iloc[index]
        prev = df.iloc[index - 1]

        with api_semaphore:
            point = mt5.symbol_info(symbol).point

        long_conditions = [
            current['close'] > max(current['ema_high'], current['ema_close'], current['ema_low']),
            prev['close'] <= max(prev['ema_high'], prev['ema_close'], prev['ema_low']),
            current['close'] < min(current['ema_outer1'], current['ema_outer2']),
            min(current['ema_outer1'], current['ema_outer2']) - current['close'] >= MIN_GAP_SECOND_STRATEGY * point,
            (current['close'] - max(current['ema_high'], current['ema_close'], current['ema_low'])) / (min(current['ema_outer1'], current['ema_outer2']) - max(current['ema_high'], current['ema_close'], current['ema_low'])) <= MAX_ZONE_PERCENTAGE
        ]

        short_conditions = [
            current['close'] < min(current['ema_high'], current['ema_close'], current['ema_low']),
            prev['close'] >= min(prev['ema_high'], prev['ema_close'], prev['ema_low']),
            current['close'] > max(current['ema_outer1'], current['ema_outer2']),
            current['close'] - max(current['ema_outer1'], current['ema_outer2']) >= MIN_GAP_SECOND_STRATEGY * point,
            (min(current['ema_high'], current['ema_close'], current['ema_low']) - current['close']) / (min(current['ema_high'], current['ema_close'], current['ema_low']) - max(current['ema_outer1'], current['ema_outer2'])) <= MAX_ZONE_PERCENTAGE
        ]

        long_condition = all(long_conditions)
        short_condition = all(short_conditions)

        logger.debug(f"{symbol} {timeframe} Secondary entry conditions: Long={long_condition}, Short={short_condition}")

        return long_condition, short_condition
    except Exception as e:
        logger.error(f"Error in check_secondary_entry for {symbol} on {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, False

def place_order(symbol, timeframe, order_type, lot_size, sl=None, tp=None):
    logger = get_logger(symbol, timeframe)
    global trades_data
    with global_lock:
        trades_data[symbol][timeframe]['analyzed'] += 1
    logger.info(f"Attempting to place {order_type} order for {symbol} on {timeframe}, lot size: {lot_size}")
    try:
        with api_semaphore:
            point = mt5.symbol_info(symbol).point
            price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Wavy Tunnel Strategy {symbol} {timeframe}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order placement failed for {symbol} on {timeframe}. Error code: {result.retcode}")
        else:
            logger.info(f"Order placed successfully for {symbol} on {timeframe}. Ticket: {result.order}")
            with global_lock:
                trades_data[symbol][timeframe]['executed'] += 1
            log_trade(symbol, timeframe, f"Trade Opened: Type={order_type}, Lot Size={lot_size}, Price={price}, SL={sl}, TP={tp}")
            log_trade_summary(symbol, timeframe)
        return result
    except Exception as e:
        logger.error(f"Error in place_order for {symbol} on {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def close_all_positions(symbol, timeframe):
    logger = get_logger(symbol, timeframe)
    global trades_data
    logger.info(f"Attempting to close all positions for {symbol} on {timeframe}")
    try:
        with api_semaphore:
            positions = mt5.positions_get(symbol=symbol)
        if positions:
            for position in positions:
                if position.type == mt5.POSITION_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(symbol).bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(symbol).ask

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": order_type,
                    "position": position.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": f"Close Wavy Tunnel Strategy {symbol} {timeframe}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                with api_semaphore:
                    result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Failed to close position {position.ticket} for {symbol} on {timeframe}. Error code: {result.retcode}")
                else:
                    logger.info(f"Position {position.ticket} closed successfully for {symbol} on {timeframe}")
                    with global_lock:
                        trades_data[symbol][timeframe]['executed'] += 1
                        profit = position.profit
                        trades_data[symbol][timeframe]['profit'] += profit
                        if profit > 0:
                            trades_data[symbol][timeframe]['won'] += 1
                        else:
                            trades_data[symbol][timeframe]['lost'] += 1
                    log_trade(symbol, timeframe, f"Trade Closed: Ticket={position.ticket}, Profit={profit:.2f}")
                    log_trade_summary(symbol, timeframe)
        else:
            logger.info(f"No open positions to close for {symbol} on {timeframe}")
    except Exception as e:
        logger.error(f"Error in close_all_positions for {symbol} on {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def run_strategy_for_symbol_timeframe(symbol, timeframe):
    logger = get_logger(symbol, timeframe)
    logger.info(f"Running Wavy Tunnel Strategy for {symbol} on {timeframe}")
    try:
        df = get_data(symbol, timeframe, 1000)
        if df is None:
            logger.warning(f"Failed to retrieve data for {symbol} on {timeframe}. Skipping this run.")
            return

        df = calculate_indicators(df, symbol, timeframe)
        if df is None:
            logger.warning(f"Failed to calculate indicators for {symbol} on {timeframe}. Skipping this run.")
            return

        primary_long, primary_short = check_primary_entry(df, -1, symbol, timeframe)
        secondary_long, secondary_short = check_secondary_entry(df, -1, symbol, timeframe)

        with api_semaphore:
            current_positions = mt5.positions_get(symbol=symbol)
        logger.info(f"Current open positions for {symbol} on {timeframe}: {len(current_positions)}")

        with global_lock:
            trades_data[symbol][timeframe]['analyzed'] += 1

        if not current_positions:
            if primary_long:
                logger.info(f"Primary long entry condition met for {symbol} on {timeframe}. Placing buy order.")
                place_order(symbol, timeframe, mt5.ORDER_TYPE_BUY, LOT_SIZE)
            elif primary_short:
                logger.info(f"Primary short entry condition met for {symbol} on {timeframe}. Placing sell order.")
                place_order(symbol, timeframe, mt5.ORDER_TYPE_SELL, LOT_SIZE)
            elif secondary_long:
                logger.info(f"Secondary long entry condition met for {symbol} on {timeframe}. Placing buy order.")
                place_order(symbol, timeframe, mt5.ORDER_TYPE_BUY, LOT_SIZE)
            elif secondary_short:
                logger.info(f"Secondary short entry condition met for {symbol} on {timeframe}. Placing sell order.")
                place_order(symbol, timeframe, mt5.ORDER_TYPE_SELL, LOT_SIZE)
            else:
                logger.info(f"No entry conditions met for {symbol} on {timeframe}. No new positions opened.")
        else:
            current = df.iloc[-1]
            for position in current_positions:
                if position.type == mt5.POSITION_TYPE_BUY:
                    if current['close'] < min(current['ema_high'], current['ema_close'], current['ema_low']):
                        logger.info(f"Exit condition met for long position in {symbol} on {timeframe}. Closing all positions.")
                        close_all_positions(symbol, timeframe)
                    else:
                        logger.info(f"No exit condition met for long position in {symbol} on {timeframe}. Maintaining position.")
                elif position.type == mt5.POSITION_TYPE_SELL:
                    if current['close'] > max(current['ema_high'], current['ema_close'], current['ema_low']):
                        logger.info(f"Exit condition met for short position in {symbol} on {timeframe}. Closing all positions.")
                        close_all_positions(symbol, timeframe)
                    else:
                        logger.info(f"No exit condition met for short position in {symbol} on {timeframe}. Maintaining position.")

        logger.info(f"Strategy run completed for {symbol} on {timeframe}.")
        log_trade_summary(symbol, timeframe)
    except Exception as e:
        logger.error(f"Error in run_strategy for {symbol} on {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def main():
    global initial_balance
    main_logger.info("Starting Multi-Pair Multi-Timeframe Wavy Tunnel Strategy main loop")

    initial_account_info = get_account_info()
    log_account_info(initial_account_info, is_initial=True)
    initial_balance = initial_account_info['balance'] if initial_account_info else 0.0

    start_time = time.time()
    end_time = start_time + 8 * 60 * 60  # 8 hours in seconds

    try:
        while time.time() < end_time:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(SYMBOLS) * len(TIMEFRAMES))) as executor:
                futures = {executor.submit(run_strategy_for_symbol_timeframe, symbol, timeframe): (symbol, timeframe)
                           for symbol in SYMBOLS
                           for timeframe in TIMEFRAMES}

                for future in concurrent.futures.as_completed(futures):
                    symbol, timeframe = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        main_logger.error(f'{symbol} {timeframe} generated an exception: {exc}')

            main_logger.info(f"Time elapsed: {(time.time() - start_time) / 60:.2f} minutes.")
            time.sleep(10)  # Wait for 10 seconds before the next iteration
    except KeyboardInterrupt:
        main_logger.info("Strategy stopped by user")
    except Exception as e:
        main_logger.error(f"Critical error in main loop: {str(e)}")
        main_logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        run_time = time.time() - start_time
        main_logger.info("Final Trade Summary:")
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                log_trade_summary(symbol, timeframe)
        main_logger.info(f"Total run time: {run_time / 60:.2f} minutes")

        final_account_info = get_account_info()
        log_account_info(final_account_info)

        main_logger.info("Exiting main loop")

if __name__ == "__main__":
    try:
        main_logger.info("Starting Multi-Pair Multi-Timeframe Wavy Tunnel Strategy script")
        main()
    except Exception as e:
        main_logger.error(f"Unhandled exception in main script: {str(e)}")
        main_logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        mt5.shutdown()
        main_logger.info("MetaTrader 5 connection shut down")
        main_logger.info("Script execution completed")