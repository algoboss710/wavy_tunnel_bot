
import numpy as np
from metatrader.data_retrieval import get_historical_data
from utils.error_handling import handle_error
from metatrader.indicators import calculate_ema
from metatrader.trade_management import place_order, close_position, modify_order
import pandas as pd
import MetaTrader5 as mt5
import logging

def calculate_ema(prices, period):
    if isinstance(prices, (float, int)):
        return prices
    elif isinstance(prices, (list, np.ndarray, pd.Series)):
        ema_values = np.full(len(prices), np.nan, dtype=np.float64)
        if len(prices) < period:
            return pd.Series(ema_values, index=prices.index)
        
        sma = np.mean(prices[:period])
        ema_values[period - 1] = sma
        multiplier = 2 / (period + 1)
        for i in range(period, len(prices)):
            ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
        return pd.Series(ema_values, index=prices.index)
    else:
        raise ValueError("Invalid input type for prices. Expected float, int, list, numpy array, or pandas Series.")

# def calculate_ema(prices, period):
#     if isinstance(prices, (float, int)):
#         return prices
#     elif isinstance(prices, (list, np.ndarray, pd.Series)):
#         ema_values = np.zeros_like(prices)
#         ema_values[:period] = np.nan
#         sma = np.mean(prices[:period])
#         ema_values[period - 1] = sma
#         multiplier = 2 / (period + 1)
#         for i in range(period, len(prices)):
#             if ema_values[i - 1] == 0:
#                 logging.error("Division by zero: ema_values[i - 1] is zero in calculate_ema")
#                 continue
#             ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
#         return pd.Series(ema_values, index=prices.index)
#     else:
#         raise ValueError("Invalid input type for prices. Expected float, int, list, numpy array, or pandas Series.")
def detect_peaks_and_dips(df, peak_type):
    peaks = []
    dips = []

    center_index = peak_type // 2

    for i in range(len(df) - peak_type):
        segment_high = df['high'].iloc[i:i + peak_type + 1].values
        segment_low = df['low'].iloc[i:i + peak_type + 1].values
        
        peak = True
        dip = True

        for j in range(peak_type + 1):
            if j != center_index:
                if segment_high[j] > segment_high[center_index]:
                    peak = False
                if segment_low[j] < segment_low[center_index]:
                    dip = False

        if peak:
            peaks.append(segment_high[center_index])
        if dip:
            dips.append(segment_low[center_index])

        # Debugging information
        print(f"Segment High: {segment_high}, Segment Low: {segment_low}, Peak: {peak}, Dip: {dip}")

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
        if threshold == 0:
            logging.error("Division by zero: threshold value is zero in check_entry_conditions")
            return False, False
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
        trade_request['profit'] = 0  # Initialize profit to 0
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

def calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, pip_value):
    risk_amount = account_balance * risk_per_trade
    if stop_loss_pips == 0 or pip_value == 0:
        logging.error("Division by zero: stop_loss_pips or pip_value is zero in calculate_position_size")
        raise ZeroDivisionError("stop_loss_pips or pip_value cannot be zero")
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return position_size

def generate_trade_signal(data, period, deviation_factor):
    upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation_factor)

    if len(upper_bound) > 0 and len(lower_bound) > 0:
        if data['close'].iloc[-1] > upper_bound.iloc[-1]:
            return 'BUY'
        elif data['close'].iloc[-1] < lower_bound.iloc[-1]:
            return 'SELL'

    return None

def adjust_deviation_factor(market_conditions):
    if market_conditions == 'volatile':
        return 2.5
    else:
        return 2.0

def run_strategy(symbols, mt5_init, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day, run_backtest=False):
    try:
        for symbol in symbols:
            start_time = pd.Timestamp.now() - pd.Timedelta(days=30)  # Example: 30 days ago
            end_time = pd.Timestamp.now()  # Current time
            data = get_historical_data(symbol, timeframe, start_time, end_time)
            if data is None:
                raise Exception(f"Failed to retrieve historical data for {symbol}")

            print(f"Historical data shape before calculations: {data.shape}")
            print(f"Historical data head before calculations:\n{data.head()}")
            print(f"Historical data for {symbol}:")
            print(data.head())
            print(f"Data types: {data.dtypes}")

            period = 20
            market_conditions = 'volatile'  # Placeholder for determining market conditions
            deviation_factor = adjust_deviation_factor(market_conditions)

            print("Calculating Wavy Tunnel indicators...")
            data['wavy_h'] = calculate_ema(data['high'], 34)
            data['wavy_c'] = calculate_ema(data['close'], 34)
            data['wavy_l'] = calculate_ema(data['low'], 34)
            data['tunnel1'] = calculate_ema(data['close'], 144)
            data['tunnel2'] = calculate_ema(data['close'], 169)
            data['long_term_ema'] = calculate_ema(data['close'], 200)
            print("Indicators calculated.")

            print("Detecting peaks and dips...")
            peak_type = 21  # Define the peak_type variable
            peaks, dips = detect_peaks_and_dips(data, peak_type)
            print(f"Peaks: {peaks[:5]}")
            print(f"Dips: {dips[:5]}")

            print(f"Historical data shape after calculations: {data.shape}")
            print(f"Historical data head after calculations:\n{data.head()}")

            print("Generating entry signals...")
            data['buy_signal'], data['sell_signal'] = zip(*data.apply(lambda x: check_entry_conditions(x, peaks, dips, symbol), axis=1))
            print("Entry signals generated.")

            if run_backtest:
                # Run backtest
                print("Running backtest...")
                # Backtest logic here
            else:
                # Run live trading
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
                    print(f"Executing BUY trade for {symbol}...")
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
                    print(f"Executing SELL trade for {symbol}...")
                    execute_trade(trade_request)

                manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

    except Exception as e:
        handle_error(e, "Failed to run the strategy")
        raise