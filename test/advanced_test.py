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
import talib as ta
import math

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

# Strategy Parameters
SYMBOLS = ["EURUSD"]  # We'll focus on a single symbol for now
TIMEFRAMES = [mt5.TIMEFRAME_M1]  # We'll focus on a single timeframe for now
POSITION_SIZE = 100
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

# Take Profit Parameters
TP_LOT_PERCENT = [60, 15, 10, 10]
TP_WEIGHTS = [0.2, 0.4, 0.6, 0.8]

# Date Range Parameters
IS_RANGE = False
FROM_DATE = datetime(2024, 2, 1)
TO_DATE = datetime(2024, 3, 16)

# Support and Resistance Proximity Parameters
IS_PROXIMITY = True
SUPPORT_RESISTANCE_PROXIMITY = 0.01

# Filter Parameters
APPLY_RSI_FILTER = False
APPLY_ATR_FILTER = False
APPLY_THRESHOLD = True

# Currency-specific parameters
THRESHOLD_USD = 2
THRESHOLD_EUR = 2
THRESHOLD_JPY = 300
THRESHOLD_GBP = 6
THRESHOLD_CHF = 2
THRESHOLD_AUD = 2

MIN_GAP_SECOND_USD = 15
MIN_GAP_SECOND_EUR = 15
MIN_GAP_SECOND_JPY = 650
MIN_GAP_SECOND_GBP = 50
MIN_GAP_SECOND_CHF = 15
MIN_GAP_SECOND_AUD = 15

LAST_TP_LIMIT_USD = 15
LAST_TP_LIMIT_EUR = 10
LAST_TP_LIMIT_JPY = 800
LAST_TP_LIMIT_GBP = 60
LAST_TP_LIMIT_CHF = 15
LAST_TP_LIMIT_AUD = 15

MIN_AUTO_TP_THRESHOLD_USD = 5
MIN_AUTO_TP_THRESHOLD_EUR = 10
MIN_AUTO_TP_THRESHOLD_JPY = 100
MIN_AUTO_TP_THRESHOLD_GBP = 50
MIN_AUTO_TP_THRESHOLD_CHF = 5
MIN_AUTO_TP_THRESHOLD_AUD = 5

# Peak detection parameter
PEAK_TYPE = 21  # This should be an odd number

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

def get_currency(symbol):
    return symbol[:3]

def get_threshold_value(symbol):
    currency = get_currency(symbol)
    thresholds = {
        "USD": THRESHOLD_USD,
        "EUR": THRESHOLD_EUR,
        "JPY": THRESHOLD_JPY,
        "GBP": THRESHOLD_GBP,
        "CHF": THRESHOLD_CHF,
        "AUD": THRESHOLD_AUD
    }
    return thresholds.get(currency, THRESHOLD_PIPS) * mt5.symbol_info(symbol).point

def get_min_gap_second(symbol):
    currency = get_currency(symbol)
    min_gaps = {
        "USD": MIN_GAP_SECOND_USD,
        "EUR": MIN_GAP_SECOND_EUR,
        "JPY": MIN_GAP_SECOND_JPY,
        "GBP": MIN_GAP_SECOND_GBP,
        "CHF": MIN_GAP_SECOND_CHF,
        "AUD": MIN_GAP_SECOND_AUD
    }
    return min_gaps.get(currency, MIN_GAP_SECOND_STRATEGY) * mt5.symbol_info(symbol).point

def get_last_tp_limit(symbol):
    currency = get_currency(symbol)
    tp_limits = {
        "USD": LAST_TP_LIMIT_USD,
        "EUR": LAST_TP_LIMIT_EUR,
        "JPY": LAST_TP_LIMIT_JPY,
        "GBP": LAST_TP_LIMIT_GBP,
        "CHF": LAST_TP_LIMIT_CHF,
        "AUD": LAST_TP_LIMIT_AUD
    }
    return tp_limits.get(currency, LAST_TP_LIMIT_USD) * mt5.symbol_info(symbol).point

def get_min_auto_tp_threshold(symbol):
    currency = get_currency(symbol)
    min_auto_tp_thresholds = {
        "USD": MIN_AUTO_TP_THRESHOLD_USD,
        "EUR": MIN_AUTO_TP_THRESHOLD_EUR,
        "JPY": MIN_AUTO_TP_THRESHOLD_JPY,
        "GBP": MIN_AUTO_TP_THRESHOLD_GBP,
        "CHF": MIN_AUTO_TP_THRESHOLD_CHF,
        "AUD": MIN_AUTO_TP_THRESHOLD_AUD
    }
    return min_auto_tp_thresholds.get(currency, MIN_AUTO_TP_THRESHOLD_USD) * mt5.symbol_info(symbol).point

def get_data(symbol, timeframe, num_bars):
    logger = get_logger(symbol, timeframe)
    logger.info(f"Fetching {num_bars} bars of {symbol} data for {timeframe} timeframe")
    try:
        with api_semaphore:
            bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        df = pd.DataFrame(bars)[['time', 'open', 'high', 'low', 'close']]
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        logger.info(f"Successfully fetched {len(df)} bars of data for {symbol} on {timeframe}")
        return df
    except Exception as e:
        logger.error(f"Error in get_data for {symbol} on {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def calculate_indicators(df):
    # EMAs
    df['wavy_h'] = ta.EMA(df['high'], timeperiod=EMA_WAVY)
    df['wavy_c'] = ta.EMA(df['close'], timeperiod=EMA_WAVY)
    df['wavy_l'] = ta.EMA(df['low'], timeperiod=EMA_WAVY)
    df['ema_12'] = ta.EMA(df['close'], timeperiod=EMA_ADDITIONAL1)
    df['tunnel1'] = ta.EMA(df['close'], timeperiod=EMA_OUTER1)
    df['tunnel2'] = ta.EMA(df['close'], timeperiod=EMA_OUTER2)
    df['longTermEMA'] = ta.EMA(df['close'], timeperiod=EMA_ADDITIONAL2)

    # RSI
    df['rsi'] = ta.RSI(df['close'], timeperiod=RSI_PERIOD)

    # ATR
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD)

    # Support and Resistance
    df['supportLevel'] = ta.SMA(ta.MIN(df['low'], timeperiod=18), timeperiod=5)
    df['resistanceLevel'] = ta.SMA(ta.MAX(df['high'], timeperiod=18), timeperiod=5)

    return df

def detect_peaks_and_dips(df):
    highs = df['high'].values
    lows = df['low'].values

    peaks = []
    dips = []

    for i in range(PEAK_TYPE // 2, len(df) - PEAK_TYPE // 2):
        is_peak = True
        is_dip = True

        for j in range(-PEAK_TYPE // 2, PEAK_TYPE // 2 + 1):
            if j == 0:
                continue
            if highs[i + j] > highs[i]:
                is_peak = False
            if lows[i + j] < lows[i]:
                is_dip = False

        if is_peak:
            peaks.append((i, highs[i]))
        if is_dip:
            dips.append((i, lows[i]))

    return peaks, dips

def find_last_peak_and_dip(peaks, dips, current_price, min_gap):
    last_peak = None
    last_dip = None

    for i, peak_price in reversed(peaks):
        if peak_price > current_price + min_gap:
            last_peak = peak_price
            break

    for i, dip_price in reversed(dips):
        if dip_price < current_price - min_gap:
            last_dip = dip_price
            break

    return last_peak, last_dip

def check_primary_entry(df, index, symbol, apply_rsi_filter, apply_atr_filter, apply_threshold):
    current = df.iloc[index]

    long_conditions = [
        current['open'] > max(current['wavy_c'], current['wavy_h'], current['wavy_l']),
        min(current['wavy_c'], current['wavy_h'], current['wavy_l']) > max(current['tunnel1'], current['tunnel2'])
    ]

    short_conditions = [
        current['open'] < min(current['wavy_c'], current['wavy_h'], current['wavy_l']),
        max(current['wavy_c'], current['wavy_h'], current['wavy_l']) < min(current['tunnel1'], current['tunnel2'])
    ]

    if apply_rsi_filter:
        long_conditions.append(current['rsi'] <= 70)
        short_conditions.append(current['rsi'] >= 30)

    if apply_atr_filter:
        atr_sma = df['atr'].rolling(ATR_PERIOD).mean().iloc[index]
        long_conditions.append(current['atr'] < 2 * atr_sma)
        short_conditions.append(current['atr'] < 2 * atr_sma)

    if apply_threshold:
        threshold_value = get_threshold_value(symbol)
        long_conditions.append(current['open'] > (max(current['wavy_c'], current['wavy_h'], current['wavy_l']) + threshold_value))
        short_conditions.append(current['open'] < (min(current['wavy_c'], current['wavy_h'], current['wavy_l']) - threshold_value))

    return all(long_conditions), all(short_conditions)

def check_secondary_entry(df, index, symbol):
    current = df.iloc[index]
    prev = df.iloc[index - 1]

    long_condition = (
        current['close'] > max(current['wavy_c'], current['wavy_h'], current['wavy_l']) and
        prev['close'] <= max(prev['wavy_c'], prev['wavy_h'], prev['wavy_l']) and
        current['close'] < min(current['tunnel1'], current['tunnel2'])
    )

    short_condition = (
        current['close'] < min(current['wavy_c'], current['wavy_h'], current['wavy_l']) and
        prev['close'] >= min(prev['wavy_c'], prev['wavy_h'], prev['wavy_l']) and
        current['close'] > max(current['tunnel1'], current['tunnel2'])
    )

    min_gap_second = get_min_gap_second(symbol)

    if long_condition:
        price_diff = min(current['tunnel1'], current['tunnel2']) - current['close']
        percentage_into_zone = (current['close'] - max(current['wavy_c'], current['wavy_h'], current['wavy_l'])) / (min(current['tunnel1'], current['tunnel2']) - max(current['wavy_c'], current['wavy_h'], current['wavy_l']))
        long_condition = long_condition and price_diff >= min_gap_second and percentage_into_zone <= MAX_ZONE_PERCENTAGE

    if short_condition:
        price_diff = current['close'] - max(current['tunnel1'], current['tunnel2'])
        percentage_into_zone = (min(current['wavy_c'], current['wavy_h'], current['wavy_l']) - current['close']) / (min(current['wavy_c'], current['wavy_h'], current['wavy_l']) - max(current['tunnel1'], current['tunnel2']))
        short_condition = short_condition and price_diff >= min_gap_second and percentage_into_zone <= MAX_ZONE_PERCENTAGE

    return long_condition, short_condition

def check_exit_conditions(df, index, position_type):
    current = df.iloc[index]

    if position_type == mt5.POSITION_TYPE_BUY:
        return current['close'] < min(current['wavy_c'], current['wavy_h'], current['wavy_l'])
    elif position_type == mt5.POSITION_TYPE_SELL:
        return current['close'] > max(current['wavy_c'], current['wavy_h'], current['wavy_l'])
    return False

def is_within_date_range(current_time, from_date, to_date, is_range_enabled):
    if not is_range_enabled:
        return True
    return from_date <= current_time <= to_date

def check_proximity(current_price, level, proximity_percentage):
    return abs(current_price - level) / level <= proximity_percentage

def is_entry_allowed(df, index, is_long, proximity_enabled, proximity_percentage):
    if not proximity_enabled:
        return True

    current = df.iloc[index]
    if is_long:
        return not check_proximity(current['open'], current['resistanceLevel'], proximity_percentage)
    else:
        return not check_proximity(current['open'], current['supportLevel'], proximity_percentage)

def calculate_take_profits(entry_price, target_price, is_long):
    direction = 1 if is_long else -1
    return [entry_price + direction * (target_price - entry_price) * weight for weight in TP_WEIGHTS]

def place_order(symbol, order_type, volume=POSITION_SIZE):
    with api_semaphore:
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Wavy Tunnel Strategy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
    return result

def place_take_profit_order(symbol, order_type, volume, tp_price, main_order):
    with api_semaphore:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if order_type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
            "position": main_order.order,
            "price": tp_price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Take Profit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
    return result

def place_order_with_take_profits(symbol, order_type, entry_price, df, index):
    current_price = entry_price
    min_auto_tp_threshold = get_min_auto_tp_threshold(symbol)
    last_tp_limit = get_last_tp_limit(symbol)

    peaks, dips = detect_peaks_and_dips(df.iloc[:index+1])
    last_peak, last_dip = find_last_peak_and_dip(peaks, dips, current_price, min_auto_tp_threshold)

    if order_type == mt5.ORDER_TYPE_BUY:
        target_price = last_peak if last_peak is not None else current_price + last_tp_limit
    else:
        target_price = last_dip if last_dip is not None else current_price - last_tp_limit

    take_profits = calculate_take_profits(entry_price, target_price, order_type == mt5.ORDER_TYPE_BUY)

    # Place the main order
    main_order = place_order(symbol, order_type, POSITION_SIZE)

    if main_order.retcode != mt5.TRADE_RETCODE_DONE:
        main_logger.error(f"Failed to place main order: {main_order.comment}")
        return

    # Place take profit orders
    for i, (tp_price, tp_percent) in enumerate(zip(take_profits, TP_LOT_PERCENT)):
        tp_size = POSITION_SIZE * (tp_percent / 100)
        tp_result = place_take_profit_order(symbol, order_type, tp_size, tp_price, main_order)
        if tp_result.retcode != mt5.TRADE_RETCODE_DONE:
            main_logger.error(f"Failed to place TP order {i+1}: {tp_result.comment}")

def get_open_positions(symbol):
    with api_semaphore:
        positions = mt5.positions_get(symbol=symbol)
    return positions if positions is not None else []

def close_position(position):
    with api_semaphore:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
            "position": position.ticket,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close Wavy Tunnel Strategy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
    return result
def run_strategy(symbol, timeframe):
    logger = get_logger(symbol, timeframe)
    logger.info(f"Running Wavy Tunnel Strategy for {symbol} on {timeframe}")

    try:
        df = get_data(symbol, timeframe, 1000)
        if df is None:
            logger.warning(f"Failed to retrieve data for {symbol} on {timeframe}. Skipping this run.")
            return

        df = calculate_indicators(df)

        wait_for_wave_touch = False

        for i in range(1, len(df)):
            current_time = df.index[i]

            if not is_within_date_range(current_time, FROM_DATE, TO_DATE, IS_RANGE):
                continue

            long_condition, short_condition = check_primary_entry(df, i, symbol, APPLY_RSI_FILTER, APPLY_ATR_FILTER, APPLY_THRESHOLD)
            second_long_condition, second_short_condition = check_secondary_entry(df, i, symbol)

            current = df.iloc[i]
            prev = df.iloc[i-1]

            # Primary strategy entry
            if not wait_for_wave_touch:
                if long_condition and is_entry_allowed(df, i, True, IS_PROXIMITY, SUPPORT_RESISTANCE_PROXIMITY):
                    logger.info(f"Long entry condition met for {symbol} on {timeframe}")
                    place_order_with_take_profits(symbol, mt5.ORDER_TYPE_BUY, current['open'], df, i)
                elif short_condition and is_entry_allowed(df, i, False, IS_PROXIMITY, SUPPORT_RESISTANCE_PROXIMITY):
                    logger.info(f"Short entry condition met for {symbol} on {timeframe}")
                    place_order_with_take_profits(symbol, mt5.ORDER_TYPE_SELL, current['open'], df, i)

            # Secondary strategy entry
            if second_long_condition:
                logger.info(f"Secondary long entry condition met for {symbol} on {timeframe}")
                place_order_with_take_profits(symbol, mt5.ORDER_TYPE_BUY, current['close'], df, i)
            elif second_short_condition:
                logger.info(f"Secondary short entry condition met for {symbol} on {timeframe}")
                place_order_with_take_profits(symbol, mt5.ORDER_TYPE_SELL, current['close'], df, i)

            # Check for exit conditions
            for position in get_open_positions(symbol):
                if check_exit_conditions(df, i, position.type):
                    logger.info(f"Exit condition met for {symbol} on {timeframe}")
                    close_result = close_position(position)
                    if close_result.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.error(f"Failed to close position: {close_result.comment}")
                    else:
                        logger.info(f"Position closed successfully for {symbol} on {timeframe}")

            # Reset wait_for_wave_touch if price touches the wave
            if wait_for_wave_touch:
                if (current['close'] >= max(current['wavy_c'], current['wavy_h'], current['wavy_l']) and
                    prev['close'] < max(prev['wavy_c'], prev['wavy_h'], prev['wavy_l'])) or \
                   (current['close'] <= min(current['wavy_c'], current['wavy_h'], current['wavy_l']) and
                    prev['close'] > min(prev['wavy_c'], prev['wavy_h'], prev['wavy_l'])):
                    wait_for_wave_touch = False
                    logger.info(f"Price touched the wave for {symbol} on {timeframe}. Resetting wait_for_wave_touch.")

            # Set wait_for_wave_touch if price is too close to support/resistance
            if IS_PROXIMITY:
                if check_proximity(current['open'], current['resistanceLevel'], SUPPORT_RESISTANCE_PROXIMITY) or \
                   check_proximity(current['open'], current['supportLevel'], SUPPORT_RESISTANCE_PROXIMITY):
                    wait_for_wave_touch = True
                    logger.info(f"Price too close to support/resistance for {symbol} on {timeframe}. Waiting for wave touch.")

        logger.info(f"Strategy run completed for {symbol} on {timeframe}")

    except Exception as e:
        logger.error(f"Error in run_strategy for {symbol} on {timeframe}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def main():
    main_logger.info("Starting Wavy Tunnel Strategy main loop")

    start_time = time.time()
    end_time = start_time + 8 * 60 * 60  # 8 hours in seconds

    try:
        while time.time() < end_time:
            for symbol in SYMBOLS:
                for timeframe in TIMEFRAMES:
                    run_strategy(symbol, timeframe)

            main_logger.info(f"Time elapsed: {(time.time() - start_time) / 60:.2f} minutes.")
            time.sleep(10)  # Wait for 10 seconds before the next iteration
    except KeyboardInterrupt:
        main_logger.info("Strategy stopped by user")
    except Exception as e:
        main_logger.error(f"Critical error in main loop: {str(e)}")
        main_logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        run_time = time.time() - start_time
        main_logger.info(f"Total run time: {run_time / 60:.2f} minutes")
        main_logger.info("Exiting main loop")

if __name__ == "__main__":
    try:
        main_logger.info("Starting Wavy Tunnel Strategy script")
        main()
    except Exception as e:
        main_logger.error(f"Unhandled exception in main script: {str(e)}")
        main_logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        mt5.shutdown()
        main_logger.info("MetaTrader 5 connection shut down")
        main_logger.info("Script execution completed")

# Additional utility functions

def calculate_position_size(symbol, risk_per_trade, stop_loss_pips):
    # This function is not used in the current implementation but kept for potential future use
    account_info = mt5.account_info()
    if account_info is None:
        raise ValueError("Failed to get account info")

    tick_size = mt5.symbol_info(symbol).trade_tick_size
    tick_value = mt5.symbol_info(symbol).trade_tick_value

    risk_amount = account_info.balance * risk_per_trade
    stop_loss_amount = stop_loss_pips * tick_size * tick_value

    if stop_loss_amount == 0:
        return POSITION_SIZE  # Return the fixed position size if we can't calculate

    position_size = risk_amount / stop_loss_amount

    # Round down to the nearest lot step
    lot_step = mt5.symbol_info(symbol).volume_step
    position_size = math.floor(position_size / lot_step) * lot_step

    # Ensure position size is within allowed limits
    min_volume = mt5.symbol_info(symbol).volume_min
    max_volume = mt5.symbol_info(symbol).volume_max
    position_size = max(min(position_size, max_volume), min_volume)

    return POSITION_SIZE  # Always return the fixed position size for now

# Additional logging functions

def log_strategy_parameters():
    main_logger.info("Strategy Parameters:")
    main_logger.info(f"Symbols: {SYMBOLS}")
    main_logger.info(f"Timeframes: {TIMEFRAMES}")
    main_logger.info(f"Position Size: {POSITION_SIZE}")
    main_logger.info(f"Account Risk: {ACCOUNT_RISK}")
    main_logger.info(f"EMA Periods: Wavy={EMA_WAVY}, Outer1={EMA_OUTER1}, Outer2={EMA_OUTER2}, Additional1={EMA_ADDITIONAL1}, Additional2={EMA_ADDITIONAL2}")
    main_logger.info(f"RSI Period: {RSI_PERIOD}")
    main_logger.info(f"ATR Period: {ATR_PERIOD}")
    main_logger.info(f"Threshold Pips: {THRESHOLD_PIPS}")
    main_logger.info(f"Min Gap Second Strategy: {MIN_GAP_SECOND_STRATEGY}")
    main_logger.info(f"Max Zone Percentage: {MAX_ZONE_PERCENTAGE}")
    main_logger.info(f"TP Lot Percent: {TP_LOT_PERCENT}")
    main_logger.info(f"TP Weights: {TP_WEIGHTS}")
    main_logger.info(f"Date Range Enabled: {IS_RANGE}")
    main_logger.info(f"From Date: {FROM_DATE}")
    main_logger.info(f"To Date: {TO_DATE}")
    main_logger.info(f"Proximity Check Enabled: {IS_PROXIMITY}")
    main_logger.info(f"Support/Resistance Proximity: {SUPPORT_RESISTANCE_PROXIMITY}")
    main_logger.info(f"Apply RSI Filter: {APPLY_RSI_FILTER}")
    main_logger.info(f"Apply ATR Filter: {APPLY_ATR_FILTER}")
    main_logger.info(f"Apply Threshold: {APPLY_THRESHOLD}")
    main_logger.info(f"Peak Type: {PEAK_TYPE}")

def log_currency_specific_parameters():
    main_logger.info("Currency-Specific Parameters:")
    main_logger.info(f"Thresholds: USD={THRESHOLD_USD}, EUR={THRESHOLD_EUR}, JPY={THRESHOLD_JPY}, GBP={THRESHOLD_GBP}, CHF={THRESHOLD_CHF}, AUD={THRESHOLD_AUD}")
    main_logger.info(f"Min Gap Second: USD={MIN_GAP_SECOND_USD}, EUR={MIN_GAP_SECOND_EUR}, JPY={MIN_GAP_SECOND_JPY}, GBP={MIN_GAP_SECOND_GBP}, CHF={MIN_GAP_SECOND_CHF}, AUD={MIN_GAP_SECOND_AUD}")
    main_logger.info(f"Last TP Limit: USD={LAST_TP_LIMIT_USD}, EUR={LAST_TP_LIMIT_EUR}, JPY={LAST_TP_LIMIT_JPY}, GBP={LAST_TP_LIMIT_GBP}, CHF={LAST_TP_LIMIT_CHF}, AUD={LAST_TP_LIMIT_AUD}")
    main_logger.info(f"Min Auto TP Threshold: USD={MIN_AUTO_TP_THRESHOLD_USD}, EUR={MIN_AUTO_TP_THRESHOLD_EUR}, JPY={MIN_AUTO_TP_THRESHOLD_JPY}, GBP={MIN_AUTO_TP_THRESHOLD_GBP}, CHF={MIN_AUTO_TP_THRESHOLD_CHF}, AUD={MIN_AUTO_TP_THRESHOLD_AUD}")

# Call these logging functions at the start of the script
log_strategy_parameters()
log_currency_specific_parameters()
# Performance tracking functions

def initialize_performance_metrics():
    return {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_profit': 0,
        'total_loss': 0,
        'max_drawdown': 0,
        'peak_balance': 0,
        'current_drawdown': 0
    }

performance_metrics = initialize_performance_metrics()

def update_performance_metrics(trade_result):
    global performance_metrics

    performance_metrics['total_trades'] += 1

    if trade_result > 0:
        performance_metrics['winning_trades'] += 1
        performance_metrics['total_profit'] += trade_result
    else:
        performance_metrics['losing_trades'] += 1
        performance_metrics['total_loss'] += abs(trade_result)

    current_balance = mt5.account_info().balance

    if current_balance > performance_metrics['peak_balance']:
        performance_metrics['peak_balance'] = current_balance
    else:
        current_drawdown = (performance_metrics['peak_balance'] - current_balance) / performance_metrics['peak_balance']
        performance_metrics['current_drawdown'] = current_drawdown
        performance_metrics['max_drawdown'] = max(performance_metrics['max_drawdown'], current_drawdown)

def log_performance_metrics():
    main_logger.info("Performance Metrics:")
    main_logger.info(f"Total Trades: {performance_metrics['total_trades']}")
    main_logger.info(f"Winning Trades: {performance_metrics['winning_trades']}")
    main_logger.info(f"Losing Trades: {performance_metrics['losing_trades']}")
    main_logger.info(f"Total Profit: {performance_metrics['total_profit']:.2f}")
    main_logger.info(f"Total Loss: {performance_metrics['total_loss']:.2f}")
    main_logger.info(f"Net Profit: {performance_metrics['total_profit'] - performance_metrics['total_loss']:.2f}")
    if performance_metrics['total_trades'] > 0:
        win_rate = (performance_metrics['winning_trades'] / performance_metrics['total_trades']) * 100
        main_logger.info(f"Win Rate: {win_rate:.2f}%")
    main_logger.info(f"Max Drawdown: {performance_metrics['max_drawdown']:.2f}%")
    main_logger.info(f"Current Drawdown: {performance_metrics['current_drawdown']:.2f}%")

# Update the close_position function to track performance
def close_position(position):
    with api_semaphore:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
            "position": position.ticket,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close Wavy Tunnel Strategy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        trade_result = result.price - position.price_open if position.type == 0 else position.price_open - result.price
        trade_result *= position.volume
        update_performance_metrics(trade_result)

    return result

# Add this to the main function to log performance metrics at the end of each run
def main():
    main_logger.info("Starting Wavy Tunnel Strategy main loop")

    start_time = time.time()
    end_time = start_time + 8 * 60 * 60  # 8 hours in seconds

    try:
        while time.time() < end_time:
            for symbol in SYMBOLS:
                for timeframe in TIMEFRAMES:
                    run_strategy(symbol, timeframe)

            main_logger.info(f"Time elapsed: {(time.time() - start_time) / 60:.2f} minutes.")
            time.sleep(10)  # Wait for 10 seconds before the next iteration
    except KeyboardInterrupt:
        main_logger.info("Strategy stopped by user")
    except Exception as e:
        main_logger.error(f"Critical error in main loop: {str(e)}")
        main_logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        run_time = time.time() - start_time
        main_logger.info(f"Total run time: {run_time / 60:.2f} minutes")
        log_performance_metrics()  # Log performance metrics at the end of the run
        main_logger.info("Exiting main loop")

# Error handling and recovery functions

def handle_mt5_error(error_code):
    error_description = mt5.last_error()
    main_logger.error(f"MetaTrader 5 error occurred. Error code: {error_code}, Description: {error_description}")

    if error_code == 10004:  # Server is busy
        main_logger.info("Server is busy. Waiting for 5 seconds before retrying.")
        time.sleep(5)
    elif error_code == 10018:  # Market is closed
        main_logger.info("Market is closed. Waiting for 1 hour before retrying.")
        time.sleep(3600)
    elif error_code in [10005, 10006]:  # Order limit reached or too many requests
        main_logger.info("Order limit reached or too many requests. Waiting for 1 minute before retrying.")
        time.sleep(60)
    else:
        main_logger.info("Unhandled error. Waiting for 30 seconds before retrying.")
        time.sleep(30)

def retry_function(func, max_attempts=3, *args, **kwargs):
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            main_logger.error(f"Error in {func.__name__}: {str(e)}")
            if attempt < max_attempts - 1:
                main_logger.info(f"Retrying {func.__name__} (attempt {attempt + 2}/{max_attempts})")
                time.sleep(5)
            else:
                main_logger.error(f"Max attempts reached for {func.__name__}. Giving up.")
                raise

# Wrap critical functions with retry mechanism
get_data = retry_function(get_data)
place_order = retry_function(place_order)
place_take_profit_order = retry_function(place_take_profit_order)
close_position = retry_function(close_position)

# Add this function to periodically check and maintain the MT5 connection
def check_mt5_connection():
    if not mt5.initialize():
        main_logger.error("Lost connection to MetaTrader 5. Attempting to reconnect...")
        attempt = 1
        while not mt5.initialize():
            main_logger.info(f"Reconnection attempt {attempt} failed. Waiting for 30 seconds before retrying.")
            time.sleep(30)
            attempt += 1
            if attempt > 10:
                main_logger.error("Failed to reconnect to MetaTrader 5 after 10 attempts. Exiting script.")
                raise ConnectionError("Failed to reconnect to MetaTrader 5")
        main_logger.info("Successfully reconnected to MetaTrader 5")

# Call this function periodically in the main loop
def main():
    main_logger.info("Starting Wavy Tunnel Strategy main loop")

    start_time = time.time()
    end_time = start_time + 8 * 60 * 60  # 8 hours in seconds

    try:
        while time.time() < end_time:
            check_mt5_connection()  # Check and maintain MT5 connection
            for symbol in SYMBOLS:
                for timeframe in TIMEFRAMES:
                    run_strategy(symbol, timeframe)

            main_logger.info(f"Time elapsed: {(time.time() - start_time) / 60:.2f} minutes.")
            time.sleep(10)  # Wait for 10 seconds before the next iteration
    except KeyboardInterrupt:
        main_logger.info("Strategy stopped by user")
    except Exception as e:
        main_logger.error(f"Critical error in main loop: {str(e)}")
        main_logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        run_time = time.time() - start_time
        main_logger.info(f"Total run time: {run_time / 60:.2f} minutes")
        log_performance_metrics()
        main_logger.info("Exiting main loop")

if __name__ == "__main__":
    try:
        main_logger.info("Starting Wavy Tunnel Strategy script")
        main()
    except Exception as e:
        main_logger.error(f"Unhandled exception in main script: {str(e)}")
        main_logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        mt5.shutdown()
        main_logger.info("MetaTrader 5 connection shut down")
        main_logger.info("Script execution completed")