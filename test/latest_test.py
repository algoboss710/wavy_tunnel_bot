import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback
import threading
import queue
import os
import talib as ta
import math
import pytz

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
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
TIMEFRAMES = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]
POSITION_SIZE = 100
ACCOUNT_RISK = 0.01
RISK_PER_TRADE = 0.01  # 1% risk per trade

# EMA Periods
EMA_WAVY = 34
EMA_OUTER1 = 144
EMA_OUTER2 = 169
EMA_ADDITIONAL1 = 12
EMA_ADDITIONAL2 = 200

# Other Parameters
RSI_PERIOD = 14
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
SUPPORT_RESISTANCE_PROXIMITY = 0.0005  # Adjusted value

# Filter Parameters
APPLY_RSI_FILTER = False
APPLY_ATR_FILTER = False
APPLY_THRESHOLD = True

# Spread parameters
ENABLE_SPREAD_CHECK = True  # New parameter to toggle spread checking

MAX_SPREAD = {
    mt5.TIMEFRAME_M1: 20,
    mt5.TIMEFRAME_M5: 15,
    mt5.TIMEFRAME_M15: 12,
    mt5.TIMEFRAME_H1: 10,
    mt5.TIMEFRAME_H4: 8,
    mt5.TIMEFRAME_D1: 6
}

CURRENCY_MAX_SPREAD = {
    "EURUSD": 30,
    "GBPUSD": 75,
    "USDJPY": 90,
    "AUDUSD": 50,
    "USDCAD": 50
}

# Peak detection parameter
PEAK_TYPE = 21  # This should be an odd number

# New parameters
MAX_TRADES_PER_DAY = 5
MIN_DISTANCE_BETWEEN_ENTRIES = 20  # in pips
TRAILING_STOP_ACTIVATION = 20  # in pips
TRAILING_STOP_DISTANCE = 10  # in pips
WAVE_TOUCH_COOLDOWN = timedelta(minutes=5)  # Cooldown period

# Volatility parameters
WARMUP_PERIOD = 100  # candles
VOLATILITY_LOOKBACK = 100  # periods for historical volatility calculation
VOLATILITY_MULTIPLIERS = {
    "EURUSD": 0.8,
    "GBPUSD": 0.85,
    "USDJPY": 0.75,
    "AUDUSD": 0.8,
    "USDCAD": 0.8
}
SL_ATR_MULTIPLIER = 2
TP_ATR_MULTIPLIER = 3

# New ATR Periods for different timeframes
ATR_PERIODS = {
    mt5.TIMEFRAME_M1: 100,
    mt5.TIMEFRAME_M5: 50,
    mt5.TIMEFRAME_M15: 30,
    mt5.TIMEFRAME_H1: 14
}

# Global variables
last_wave_touch_time = datetime.min.replace(tzinfo=timezone.utc)
last_recalibration_time = datetime.min.replace(tzinfo=timezone.utc)
RECALIBRATION_INTERVAL = timedelta(days=7)  # Recalibrate weekly

def get_logger(symbol, timeframe):
    key = f"{symbol}_{get_timeframe_name(timeframe)}"
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

def get_currency(symbol):
    return symbol[:3]

def get_threshold_value(symbol):
    currency = get_currency(symbol)
    thresholds = {
        "USD": THRESHOLD_PIPS,
        "EUR": THRESHOLD_PIPS,
        "JPY": THRESHOLD_PIPS,
        "GBP": THRESHOLD_PIPS,
        "AUD": THRESHOLD_PIPS,
        "CAD": THRESHOLD_PIPS
    }
    return thresholds.get(currency, THRESHOLD_PIPS) * mt5.symbol_info(symbol).point

def get_min_gap_second(symbol):
    currency = get_currency(symbol)
    min_gaps = {
        "USD": MIN_GAP_SECOND_STRATEGY,
        "EUR": MIN_GAP_SECOND_STRATEGY,
        "JPY": MIN_GAP_SECOND_STRATEGY,
        "GBP": MIN_GAP_SECOND_STRATEGY,
        "AUD": MIN_GAP_SECOND_STRATEGY,
        "CAD": MIN_GAP_SECOND_STRATEGY
    }
    return min_gaps.get(currency, MIN_GAP_SECOND_STRATEGY) * mt5.symbol_info(symbol).point

def get_last_tp_limit(symbol):
    currency = get_currency(symbol)
    tp_limits = {
        "USD": 15,
        "EUR": 10,
        "JPY": 800,
        "GBP": 60,
        "AUD": 15,
        "CAD": 15
    }
    return tp_limits.get(currency, 15) * mt5.symbol_info(symbol).point

def get_min_auto_tp_threshold(symbol):
    currency = get_currency(symbol)
    min_auto_tp_thresholds = {
        "USD": 5,
        "EUR": 10,
        "JPY": 100,
        "GBP": 50,
        "AUD": 5,
        "CAD": 5
    }
    return min_auto_tp_thresholds.get(currency, 5) * mt5.symbol_info(symbol).point

def get_timeframe_name(timeframe):
    timeframe_map = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1"
    }
    return timeframe_map.get(timeframe, str(timeframe))

def get_seconds_for_timeframe(timeframe):
    timeframe_seconds = {
        mt5.TIMEFRAME_M1: 60,
        mt5.TIMEFRAME_M5: 300,
        mt5.TIMEFRAME_M15: 900,
        mt5.TIMEFRAME_H1: 3600,
        mt5.TIMEFRAME_H4: 14400,
        mt5.TIMEFRAME_D1: 86400
    }
    return timeframe_seconds.get(timeframe, 60)  # Default to 60 seconds if unknown timeframe

def get_data(symbol, timeframe, num_bars):
    logger = get_logger(symbol, timeframe)
    current_time = datetime.now()
    last_fetch_time = getattr(get_data, f'last_fetch_time_{symbol}_{timeframe}', None)

    if last_fetch_time is None or (current_time - last_fetch_time).total_seconds() > get_seconds_for_timeframe(timeframe) * 0.8:
        logger.info(f"Fetching {num_bars} bars of {symbol} data for {get_timeframe_name(timeframe)} timeframe")
        try:
            with api_semaphore:
                bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
            df = pd.DataFrame(bars)[['time', 'open', 'high', 'low', 'close']]
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            setattr(get_data, f'last_fetch_time_{symbol}_{timeframe}', current_time)
            setattr(get_data, f'cached_data_{symbol}_{timeframe}', df)
            logger.info(f"Successfully fetched {len(df)} bars of data for {symbol} on {get_timeframe_name(timeframe)}")
            return df
        except Exception as e:
            logger.error(f"Error in get_data for {symbol} on {get_timeframe_name(timeframe)}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    else:
        logger.info(f"Using cached data for {symbol} on {get_timeframe_name(timeframe)}")
        return getattr(get_data, f'cached_data_{symbol}_{timeframe}', None)

def is_market_open(symbol):
    main_logger.info(f"Checking if market is open for {symbol}")
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        main_logger.error(f"Failed to get symbol info for {symbol}")
        return False

    if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
        main_logger.warning(f"{symbol} is not available for full trading")
        return False

    last_tick = mt5.symbol_info_tick(symbol)
    if last_tick is None:
        main_logger.error(f"Failed to get last tick for {symbol}")
        return False

    current_time = datetime.now(timezone.utc)
    tick_time = datetime.fromtimestamp(last_tick.time, tz=timezone.utc)
    if (current_time - tick_time).total_seconds() > 60:
        main_logger.warning(f"Last tick for {symbol} is more than 60 seconds old")
        return False

    main_logger.info(f"Market is open for {symbol}")
    return True

def log_strategy_parameters():
    main_logger.info("Strategy Parameters:")
    main_logger.info(f"SYMBOLS: {SYMBOLS}")
    main_logger.info(f"TIMEFRAMES: {[get_timeframe_name(tf) for tf in TIMEFRAMES]}")
    main_logger.info(f"POSITION_SIZE: {POSITION_SIZE}")
    main_logger.info(f"ACCOUNT_RISK: {ACCOUNT_RISK}")
    main_logger.info(f"EMA Periods: WAVY={EMA_WAVY}, OUTER1={EMA_OUTER1}, OUTER2={EMA_OUTER2}")
    main_logger.info(f"RSI_PERIOD: {RSI_PERIOD}")
    main_logger.info(f"ATR_PERIODS: {ATR_PERIODS}")
    main_logger.info(f"THRESHOLD_PIPS: {THRESHOLD_PIPS}")
    main_logger.info(f"MIN_GAP_SECOND_STRATEGY: {MIN_GAP_SECOND_STRATEGY}")
    main_logger.info(f"MAX_ZONE_PERCENTAGE: {MAX_ZONE_PERCENTAGE}")
    main_logger.info(f"VOLATILITY_LOOKBACK: {VOLATILITY_LOOKBACK}")
    main_logger.info(f"VOLATILITY_MULTIPLIERS: {VOLATILITY_MULTIPLIERS}")
    main_logger.info(f"SL_ATR_MULTIPLIER: {SL_ATR_MULTIPLIER}")
    main_logger.info(f"TP_ATR_MULTIPLIER: {TP_ATR_MULTIPLIER}")
    main_logger.info(f"TRAILING_STOP_ACTIVATION: {TRAILING_STOP_ACTIVATION}")
    main_logger.info(f"TRAILING_STOP_DISTANCE: {TRAILING_STOP_DISTANCE}")
    main_logger.info(f"ENABLE_SPREAD_CHECK: {ENABLE_SPREAD_CHECK}")
    main_logger.info(f"MAX_SPREAD: {MAX_SPREAD}")
    main_logger.info(f"CURRENCY_MAX_SPREAD: {CURRENCY_MAX_SPREAD}")

def calculate_indicators(df, timeframe):
    df['wavy_h'] = ta.EMA(df['high'], timeperiod=EMA_WAVY)
    df['wavy_c'] = ta.EMA(df['close'], timeperiod=EMA_WAVY)
    df['wavy_l'] = ta.EMA(df['low'], timeperiod=EMA_WAVY)
    df['ema_12'] = ta.EMA(df['close'], timeperiod=EMA_ADDITIONAL1)
    df['tunnel1'] = ta.EMA(df['close'], timeperiod=EMA_OUTER1)
    df['tunnel2'] = ta.EMA(df['close'], timeperiod=EMA_OUTER2)
    df['longTermEMA'] = ta.EMA(df['close'], timeperiod=EMA_ADDITIONAL2)
    df['rsi'] = ta.RSI(df['close'], timeperiod=RSI_PERIOD)
    atr_period = ATR_PERIODS.get(timeframe, 14)
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    df['atr_normalized'] = df['atr'] / df['close'] * 10000
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
        atr_sma = df['atr'].rolling(ATR_PERIODS.get(df.index.freq, 14)).mean().iloc[index]
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

def calculate_volatility_percentile(df, lookback=VOLATILITY_LOOKBACK):
    recent_volatility = df['atr_normalized'].rolling(window=lookback).mean()
    return max(recent_volatility.quantile(0.25), 1e-8)  # Ensure non-zero value

def calculate_dynamic_threshold(df, symbol, timeframe, lookback=500):
    hist_volatility = df['atr_normalized'].rolling(window=lookback).mean()
    return hist_volatility.mean() * VOLATILITY_MULTIPLIERS.get(symbol, 0.8)

def check_volatility(df, index, symbol, timeframe):
    if df is None or 'atr_normalized' not in df.columns or index < VOLATILITY_LOOKBACK:
        return 0, False
    current_volatility = df['atr_normalized'].iloc[index]
    volatility_percentile = calculate_volatility_percentile(df.iloc[:index+1])
    dynamic_threshold = calculate_dynamic_threshold(df.iloc[:index+1], symbol, timeframe)
    volatility_ratio = current_volatility / volatility_percentile if volatility_percentile > 0 else 0
    is_volatile_enough = volatility_ratio >= 1 and current_volatility >= dynamic_threshold
    return volatility_ratio, is_volatile_enough

def check_volatility_breakout(df, index, lookback=20):
    recent_atr = df['atr_normalized'].iloc[index-lookback:index]
    atr_mean = recent_atr.mean()
    atr_std = recent_atr.std()
    breakout_threshold = atr_mean + 2 * atr_std
    current_atr = df['atr_normalized'].iloc[index]
    return current_atr > breakout_threshold

def is_entry_allowed(df, index, is_long, proximity_enabled, proximity_percentage, symbol, timeframe):
    global last_wave_touch_time
    logger = get_logger(symbol, get_timeframe_name(timeframe))
    if df is None or len(df) == 0:
        logger.error("DataFrame is None or empty in is_entry_allowed")
        return False
    if index < 0 or index >= len(df):
        logger.error(f"Invalid index {index} for DataFrame of length {len(df)}")
        return False
    try:
        current_time = df.index[index].replace(tzinfo=timezone.utc)
        if last_wave_touch_time.tzinfo is None:
            last_wave_touch_time = last_wave_touch_time.replace(tzinfo=timezone.utc)
        if current_time - last_wave_touch_time < WAVE_TOUCH_COOLDOWN:
            logger.info(f"Entry not allowed. Still in cooldown period. Time since last touch: {current_time - last_wave_touch_time}")
            return False
        current = df.iloc[index]
        if is_long:
            proximity = check_proximity(current['open'], current['resistanceLevel'], proximity_percentage)
            logger.info(f"Long entry proximity check: Current={current['open']:.5f}, Resistance={current['resistanceLevel']:.5f}, Proximity={proximity}")
            return not proximity
        else:
            proximity = check_proximity(current['open'], current['supportLevel'], proximity_percentage)
            logger.info(f"Short entry proximity check: Current={current['open']:.5f}, Support={current['supportLevel']:.5f}, Proximity={proximity}")
            return not proximity
    except Exception as e:
        logger.error(f"Error in is_entry_allowed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def calculate_take_profits(entry_price, target_price, is_long):
    direction = 1 if is_long else -1
    return [entry_price + direction * (target_price - entry_price) * weight for weight in TP_WEIGHTS]

def check_spread(symbol, timeframe):
    if not ENABLE_SPREAD_CHECK:
        return True, f"Spread check disabled for {symbol} on {get_timeframe_name(timeframe)}"

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, f"Unable to get tick info for {symbol}"

    spread = (tick.ask - tick.bid) / mt5.symbol_info(symbol).point
    max_allowed_spread_timeframe = MAX_SPREAD.get(timeframe, max(MAX_SPREAD.values()))
    max_allowed_spread_currency = CURRENCY_MAX_SPREAD.get(symbol, max(CURRENCY_MAX_SPREAD.values()))
    max_allowed_spread = min(max_allowed_spread_timeframe, max_allowed_spread_currency)

    timeframe_name = get_timeframe_name(timeframe)
    spread_info = f"Current spread for {symbol} on {timeframe_name}: {spread:.1f} pips. Max allowed: {max_allowed_spread} pips"

    return spread <= max_allowed_spread, spread_info

def calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, volatility_ratio):
    base_position = (account_balance * risk_per_trade) / stop_loss_pips
    return base_position * min(volatility_ratio, 2)  # Cap at 2x base position

def calculate_sl_tp(entry_price, atr, direction):
    sl_distance = atr * SL_ATR_MULTIPLIER
    tp_distance = atr * TP_ATR_MULTIPLIER
    if direction == 'long':
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
    return sl, tp

def place_order(symbol, order_type, volume):
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
        if result is None:
            main_logger.error(f"Failed to place order: result is None. Last error: {mt5.last_error()}")
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
            main_logger.error(f"Failed to place order. Error code: {result.retcode}, Description: {result.comment}")
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

    main_order = place_order(symbol, order_type, POSITION_SIZE)

    if main_order is None or main_order.retcode != mt5.TRADE_RETCODE_DONE:
        main_logger.error(f"Failed to place main order: {mt5.last_error() if main_order is None else main_order.comment}")
        return

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

def calculate_risk_reward_ratio(entry_price, stop_loss, take_profit, order_type):
    if order_type == mt5.ORDER_TYPE_BUY:
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
    return reward / risk if risk != 0 else 0

def get_last_trade_time(symbol):
    orders = mt5.history_orders_get(symbol=symbol, timeframe=mt5.TIMEFRAME_D1)
    if orders:
        return max(order.time_setup for order in orders)
    return datetime.min.replace(tzinfo=timezone.utc)

def place_order_with_sl_tp(symbol, order_type, volume, price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "Wavy Tunnel Strategy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        main_logger.error(f"Failed to place order with SL/TP. Error code: {result.retcode}, Description: {result.comment}")
    return result

def modify_order_sl_tp(ticket, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": sl,
        "tp": tp,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        main_logger.error(f"Failed to modify SL/TP. Error code: {result.retcode}, Description: {result.comment}")
    return result

def run_strategy(symbol, timeframe):
    logger = get_logger(symbol, timeframe)
    logger.info(f"Running strategy for {symbol} on {get_timeframe_name(timeframe)}")

    try:
        df = get_data(symbol, timeframe, 1000 + WARMUP_PERIOD)
        if df is None or len(df) < WARMUP_PERIOD + 100:
            logger.warning(f"Insufficient data for {symbol} on {get_timeframe_name(timeframe)}. Skipping this run.")
            return

        df = calculate_indicators(df, timeframe)
        df = df.iloc[WARMUP_PERIOD:]  # Skip the warmup period for actual trading

        i = -1  # Index of the most recent candle
        current = df.iloc[i]

        logger.info(f"Current price for {symbol}: {current['close']}")
        logger.info(f"EMA values - Wavy High: {current['wavy_h']:.5f}, Wavy Close: {current['wavy_c']:.5f}, Wavy Low: {current['wavy_l']:.5f}")
        logger.info(f"Tunnel values - Tunnel1: {current['tunnel1']:.5f}, Tunnel2: {current['tunnel2']:.5f}")
        logger.info(f"RSI: {current['rsi']:.2f}, ATR: {current['atr']:.5f}")

        spread_ok, spread_info = check_spread(symbol, timeframe)
        logger.info(spread_info)
        if ENABLE_SPREAD_CHECK and not spread_ok:
            logger.info("Trade not executed due to high spread.")
            return

        volatility_ratio, is_volatile_enough = check_volatility(df, i, symbol, timeframe)
        is_breakout = check_volatility_breakout(df, i)

        logger.info(f"Volatility ratio: {volatility_ratio:.2f}, Is volatile enough: {is_volatile_enough}")
        logger.info(f"Volatility breakout: {is_breakout}")

        if not is_volatile_enough and not is_breakout:
            logger.info("Trade not executed due to low volatility and no breakout.")
            return

        long_condition, short_condition = check_primary_entry(df, i, symbol, APPLY_RSI_FILTER, APPLY_ATR_FILTER, APPLY_THRESHOLD)
        logger.info(f"Primary entry conditions: Long={long_condition}, Short={short_condition}")

        if long_condition and is_entry_allowed(df, i, True, IS_PROXIMITY, SUPPORT_RESISTANCE_PROXIMITY, symbol, timeframe):
            entry_price = current['close']
            atr = df['atr'].iloc[i]
            sl, tp = calculate_sl_tp(entry_price, atr, 'long')
            position_size = calculate_position_size(mt5.account_info().balance, RISK_PER_TRADE, abs(entry_price - sl), volatility_ratio)

            risk_reward_ratio = calculate_risk_reward_ratio(entry_price, sl, tp, mt5.ORDER_TYPE_BUY)
            logger.info(f"Long trade consideration - Entry: {entry_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}, R:R ratio: {risk_reward_ratio:.2f}")

            logger.info(f"Attempting to place long order for {symbol} on {get_timeframe_name(timeframe)}")
            result = place_order_with_sl_tp(symbol, mt5.ORDER_TYPE_BUY, position_size, entry_price, sl, tp)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Long order placed successfully: Ticket={result.order}")
                update_performance_metrics(result.profit)
            else:
                logger.error(f"Failed to place long order. Error code: {result.retcode}, Description: {result.comment}")

        elif short_condition and is_entry_allowed(df, i, False, IS_PROXIMITY, SUPPORT_RESISTANCE_PROXIMITY, symbol, timeframe):
            entry_price = current['close']
            atr = df['atr'].iloc[i]
            sl, tp = calculate_sl_tp(entry_price, atr, 'short')
            position_size = calculate_position_size(mt5.account_info().balance, RISK_PER_TRADE, abs(entry_price - sl), volatility_ratio)

            risk_reward_ratio = calculate_risk_reward_ratio(entry_price, sl, tp, mt5.ORDER_TYPE_SELL)
            logger.info(f"Short trade consideration - Entry: {entry_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}, R:R ratio: {risk_reward_ratio:.2f}")

            logger.info(f"Attempting to place short order for {symbol} on {get_timeframe_name(timeframe)}")
            result = place_order_with_sl_tp(symbol, mt5.ORDER_TYPE_SELL, position_size, entry_price, sl, tp)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Short order placed successfully: Ticket={result.order}")
                update_performance_metrics(result.profit)
            else:
                logger.error(f"Failed to place short order. Error code: {result.retcode}, Description: {result.comment}")

        else:
            logger.info("No trade executed. Entry conditions not met or entry not allowed due to proximity.")

        open_positions = get_open_positions(symbol)
        logger.info(f"Open positions for {symbol}: {len(open_positions)}")
        for position in open_positions:
            current_profit = position.profit
            logger.info(f"Position {position.ticket}: Type={position.type}, Volume={position.volume}, Open Price={position.price_open}, Current Profit={current_profit}")

            if check_exit_conditions(df, i, position.type):
                logger.info(f"Exit condition met for position {position.ticket}")
                close_result = close_position(position)
                if close_result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Closed position {position.ticket}. Profit: {close_result.profit}")
                    update_performance_metrics(close_result.profit)
                else:
                    logger.error(f"Failed to close position {position.ticket}. Error: {close_result.comment}")

            if position.type == mt5.POSITION_TYPE_BUY:
                if current['close'] - position.price_open >= TRAILING_STOP_ACTIVATION * mt5.symbol_info(symbol).point:
                    new_sl = current['close'] - TRAILING_STOP_DISTANCE * mt5.symbol_info(symbol).point
                    if new_sl > position.sl:
                        logger.info(f"Updating trailing stop for long position {position.ticket}. New SL: {new_sl}")
                        modify_result = modify_order_sl_tp(position.ticket, new_sl, position.tp)
                        if modify_result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"Updated trailing stop for position {position.ticket}. New SL: {new_sl}")
                        else:
                            logger.error(f"Failed to update trailing stop for position {position.ticket}. Error: {modify_result.comment}")

            elif position.type == mt5.POSITION_TYPE_SELL:
                if position.price_open - current['close'] >= TRAILING_STOP_ACTIVATION * mt5.symbol_info(symbol).point:
                    new_sl = current['close'] + TRAILING_STOP_DISTANCE * mt5.symbol_info(symbol).point
                    if new_sl < position.sl:
                        logger.info(f"Updating trailing stop for short position {position.ticket}. New SL: {new_sl}")
                        modify_result = modify_order_sl_tp(position.ticket, new_sl, position.tp)
                        if modify_result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"Updated trailing stop for position {position.ticket}. New SL: {new_sl}")
                        else:
                            logger.error(f"Failed to update trailing stop for position {position.ticket}. Error: {modify_result.comment}")

    except Exception as e:
        logger.error(f"Error in run_strategy for {symbol} on {get_timeframe_name(timeframe)}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def log_summary(symbols, timeframes):
    main_logger.info("--- Market Summary ---")
    for symbol in symbols:
        for timeframe in timeframes:
            spread_ok, spread_info = check_spread(symbol, timeframe)
            df = get_data(symbol, timeframe, 100 + WARMUP_PERIOD)
            if df is not None and len(df) > WARMUP_PERIOD:
                df = df.iloc[WARMUP_PERIOD:]
                volatility_ratio, is_volatile_enough = check_volatility(df, -1, symbol, timeframe)
                is_breakout = check_volatility_breakout(df, -1)
                main_logger.info(f"{symbol} {get_timeframe_name(timeframe)}: Spread OK: {spread_ok}, Volatility Ratio: {volatility_ratio:.2f}, Breakout: {is_breakout}")
            else:
                main_logger.info(f"{symbol} {get_timeframe_name(timeframe)}: Unable to fetch data")
    main_logger.info("----------------------")

def log_mt5_timeframes():
    main_logger.info("MetaTrader 5 Timeframes:")
    for name, value in mt5.__dict__.items():
        if name.startswith("TIMEFRAME_"):
            main_logger.info(f"{name}: {value}")

def main():
    main_logger.info("Starting Wavy Tunnel Strategy main loop")
    main_logger.info("Logging MetaTrader 5 timeframes...")
    log_mt5_timeframes()
    main_logger.info("Logging strategy parameters...")
    log_strategy_parameters()

    start_time = time.time()
    end_time = start_time + 8 * 60 * 60  # 8 hours in seconds
    last_summary_time = start_time
    last_market_check_time = start_time

    main_logger.info("Entering main loop...")

    try:
        while time.time() < end_time:
            current_time = time.time()
            main_logger.info(f"Current time: {datetime.fromtimestamp(current_time)}")

            # Check market status every 30 minutes
            if current_time - last_market_check_time >= 1800:  # 1800 seconds = 30 minutes
                main_logger.info("Checking market status...")
                market_open = all(is_market_open(symbol) for symbol in SYMBOLS)
                if not market_open:
                    main_logger.warning("Market is closed for one or more symbols. Waiting for 5 minutes before next check.")
                    time.sleep(300)  # Wait for 5 minutes before next iteration
                    last_market_check_time = current_time
                    continue
                last_market_check_time = current_time

            main_logger.info("Running strategies for all symbols and timeframes...")
            for symbol in SYMBOLS:
                for timeframe in TIMEFRAMES:
                    main_logger.info(f"Running strategy for {symbol} on {get_timeframe_name(timeframe)}")
                    run_strategy(symbol, timeframe)
                    time.sleep(1)  # Add a small delay between each run to prevent overloading

            # Log summary every 15 minutes
            if current_time - last_summary_time >= 900:  # 900 seconds = 15 minutes
                main_logger.info("Logging summary...")
                log_summary(SYMBOLS, TIMEFRAMES)
                last_summary_time = current_time

            main_logger.info("Waiting for 10 seconds before next iteration...")
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
        current_drawdown = (performance_metrics['peak_balance'] - current_balance) / performance_metrics['peak_balance'] * 100
        performance_metrics['current_drawdown'] = current_drawdown
        if current_drawdown > performance_metrics['max_drawdown']:
            performance_metrics['max_drawdown'] = current_drawdown

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