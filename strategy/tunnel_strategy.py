import numpy as np
from metatrader.data_retrieval import get_historical_data
from utils.error_handling import handle_error
from metatrader.indicators import calculate_ema
from metatrader.trade_management import place_order, close_position, modify_order
import pandas as pd
import MetaTrader5 as mt5

def calculate_ema(data, period):
    ema_values = [None] * (period - 1)  # Initialize with None for the first period - 1 values
    sma = sum(data[:period]) / period
    ema_values.append(sma)
    multiplier = 2 / (period + 1)
    for price in data[period:]:
        new_ema = (price - ema_values[-1]) * multiplier + ema_values[-1]
        ema_values.append(new_ema)
    
    # Ensure the length matches the input data
    while len(ema_values) < len(data):
        ema_values.append(None)
    
    return ema_values

def detect_peaks_and_dips(df, peak_type):
    peaks = []
    dips = []
    for i in range(len(df)):
        is_peak = True
        is_dip = True
        for j in range(peak_type):
            start_index = max(0, i - j)
            end_index = min(len(df) - 1, i + j)
            if df['high'][i] <= df['high'][start_index] or df['high'][i] <= df['high'][end_index]:
                is_peak = False
            if df['low'][i] >= df['low'][start_index] or df['low'][i] >= df['low'][end_index]:
                is_dip = False
        if is_peak:
            peaks.append(df['high'][i])
        if is_dip:
            dips.append(df['low'][i])
    return peaks, dips

def check_entry_conditions(row, peaks, dips, symbol):
    buy_condition = (
        row['close'] > max(row['wavy_c'], row['wavy_h'], row['wavy_l']) and
        min(row['wavy_c'], row['wavy_h'], row['wavy_l']) > max(row['tunnel1'], row['tunnel2']) and
        row['close'] in peaks  # Check if the current close price is a peak
    )
    sell_condition = (
        row['close'] < min(row['wavy_c'], row['wavy_h'], row['wavy_l']) and
        max(row['wavy_c'], row['wavy_h'], row['wavy_l']) < min(row['tunnel1'], row['tunnel2']) and
        row['close'] in dips  # Check if the current close price is a dip
    )
    threshold_values = {
        'USD': 2,
        'EUR': 2,
        'JPY': 300,
        'GBP': 6,
        'CHF': 2,
        'AUD': 2,
        'default': 100
    }
    apply_threshold = True
    if apply_threshold:
        threshold = threshold_values.get(symbol[:3], threshold_values['default']) * mt5.symbol_info(symbol).trade_tick_size
        buy_condition &= row['close'] > max(row['wavy_c'], row['wavy_h'], row['wavy_l']) + threshold
        sell_condition &= row['close'] < min(row['wavy_c'], row['wavy_h'], row['wavy_l']) - threshold
    return buy_condition, sell_condition

def execute_trade(trade_request):
    try:
        result = place_order(
            trade_request['symbol'],
            trade_request['action'].lower(),
            trade_request['volume'],
            trade_request['price'],
            trade_request['sl'],
            trade_request['tp']
        )
        if result == 'Order failed':
            raise Exception("Failed to execute trade")
        return result
    except Exception as e:
        handle_error(e, "Failed to execute trade")
        return None

def manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for position in positions:
                if position.profit >= min_take_profit:
                    close_position(position.ticket)
                elif position.profit <= -max_loss_per_day:
                    close_position(position.ticket)
                else:
                    current_equity = mt5.account_info().equity
                    if current_equity <= starting_equity * 0.9:  # Close position if equity drops by 10%
                        close_position(position.ticket)
                    elif mt5.positions_total() >= max_trades_per_day:
                        close_position(position.ticket)
    except Exception as e:
        handle_error(e, "Failed to manage position")

def calculate_tunnel_bounds(data, period, deviation_factor):
    ema = calculate_ema(data['close'], period)
    volatility = np.std(data['close'])
    deviation = deviation_factor * volatility
    upper_bound = ema + deviation
    lower_bound = ema - deviation
    return upper_bound, lower_bound

def calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value):
    # Calculate position size based on balance and risk percentage
    risk_amount = balance * risk_percent
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return position_size

def generate_trade_signal(data, period, deviation_factor):
    upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation_factor)
    if data['close'].iloc[-1] > upper_bound[-1]:
        return 'BUY'
    elif data['close'].iloc[-1] < lower_bound[-1]:
        return 'SELL'
    else:
        return None

def adjust_deviation_factor(market_conditions):
    if market_conditions == 'volatile':
        return 2.5
    else:
        return 2.0

def run_strategy(symbols, mt5_init, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    try:
        for symbol in symbols:
            start_time = pd.Timestamp.now() - pd.Timedelta(days=30)  # Example: 30 days ago
            end_time = pd.Timestamp.now()  # Current time
            data = get_historical_data(symbol, timeframe, start_time, end_time)
            if data is None:
                raise Exception(f"Failed to retrieve historical data for {symbol}")

            period = 20
            market_conditions = 'volatile'  # Placeholder for determining market conditions
            deviation_factor = adjust_deviation_factor(market_conditions)

            signal = generate_trade_signal(data, period, deviation_factor)

            if signal == 'BUY':
                trade_request = {
                    'action': 'BUY',
                    'symbol': symbol,
                    'volume': lot_size,
                    'price': data['close'].iloc[-1],
                    'sl': data['close'].iloc[-1] - (1.5 * np.std(data['close'])),
                    'tp': data['close'].iloc[-1] + (2 * np.std(data['close'])),
                    'deviation': 10,
                    'magic': 12345,
                    'comment': 'Tunnel Strategy',
                    'type': 'ORDER_TYPE_BUY',
                    'type_filling': 'ORDER_FILLING_FOK',
                    'type_time': 'ORDER_TIME_GTC'
                }
                execute_trade(trade_request)
            elif signal == 'SELL':
                trade_request = {
                    'action': 'SELL',
                    'symbol': symbol,
                    'volume': lot_size,
                    'price': data['close'].iloc[-1],
                    'sl': data['close'].iloc[-1] + (1.5 * np.std(data['close'])),
                    'tp': data['close'].iloc[-1] - (2 * np.std(data['close'])),
                    'deviation': 10,
                    'magic': 12345,
                    'comment': 'Tunnel Strategy',
                    'type': 'ORDER_TYPE_SELL',
                    'type_filling': 'ORDER_FILLING_FOK',
                    'type_time': 'ORDER_TIME_GTC'
                }
                execute_trade(trade_request)

            manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

    except Exception as e:
        handle_error(e, "Failed to run the strategy")
