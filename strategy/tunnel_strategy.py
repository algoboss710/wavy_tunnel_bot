
import pandas as pd
import numpy as np 
import MetaTrader5 as mt5
import logging
from metatrader.data_retrieval import get_historical_data
from utils.error_handling import handle_error
from metatrader.indicators import calculate_ema
from metatrader.trade_management import place_order, close_position, modify_order

def calculate_ema(prices, period):
    if not isinstance(prices, (list, np.ndarray, pd.Series)):
        raise ValueError("Invalid input type for prices. Expected list, numpy array, or pandas Series.")
    
    logging.debug(f"Calculating EMA for period: {period}, prices: {prices}")
    
    prices = pd.Series(prices)
    prices = pd.to_numeric(prices, errors='coerce')
    logging.debug(f"Prices converted to numeric: {prices}")

    ema_values = np.full(len(prices), np.nan, dtype=np.float64)
    if len(prices) < period:
        return pd.Series(ema_values, index=prices.index)
    
    sma = np.mean(prices[:period])
    ema_values[period - 1] = sma
    logging.debug(f"Initial SMA: {sma}")
    
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
        logging.debug(f"EMA value at index {i}: {ema_values[i]}")
    
    return pd.Series(ema_values, index=prices.index)

def detect_peaks_and_dips(df, peak_type):
    if not np.issubdtype(df['high'].dtype, np.number) or not np.issubdtype(df['low'].dtype, np.number):
        raise TypeError("High and Low columns must contain numeric data.")

    logging.debug(f"Detecting peaks and dips with peak_type: {peak_type}")
    
    highs = df['high'].values
    lows = df['low'].values
    center_index = peak_type // 2
    peaks = []
    dips = []
    
    for i in range(center_index, len(highs) - center_index):
        peak_window = highs[i - center_index:i + center_index + 1]
        dip_window = lows[i - center_index:i + center_index + 1]
        
        if all(peak_window[center_index] > peak_window[j] for j in range(len(peak_window)) if j != center_index):
            peaks.append(highs[i])
        
        if all(dip_window[center_index] < dip_window[j] for j in range(len(dip_window)) if j != center_index):
            dips.append(lows[i])
    
    logging.debug(f"Detected peaks: {peaks}")
    logging.debug(f"Detected dips: {dips}")
    
    return peaks, dips

def check_entry_conditions(row, peaks, dips, symbol):
    logging.debug(f"Checking entry conditions for row: {row}")
    logging.debug(f"Peaks: {peaks}")
    logging.debug(f"Dips: {dips}")

    wavy_c, wavy_h, wavy_l = row['wavy_c'], row['wavy_h'], row['wavy_l']
    tunnel1, tunnel2 = row['tunnel1'], row['tunnel2']
    close_price = row['close']

    logging.debug(f"wavy_c: {wavy_c}, wavy_h: {wavy_h}, wavy_l: {wavy_l}")
    logging.debug(f"tunnel1: {tunnel1}, tunnel2: {tunnel2}")
    logging.debug(f"close_price: {close_price}")

    buy_condition = (
        close_price > max(wavy_c, wavy_h, wavy_l) and
        min(wavy_c, wavy_h, wavy_l) > max(tunnel1, tunnel2) and
        close_price in peaks  # Check if the current close price is a peak
    )
    sell_condition = (
        close_price < min(wavy_c, wavy_h, wavy_l) and
        max(wavy_c, wavy_h, wavy_l) < min(tunnel1, tunnel2) and
        close_price in dips  # Check if the current close price is a dip
    )
    
    logging.debug(f"Initial Buy condition: {buy_condition}")
    logging.debug(f"Initial Sell condition: {sell_condition}")

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
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logging.error(f"Failed to get symbol info for {symbol}")
            return False, False
        
        threshold = threshold_values.get(symbol[:3], threshold_values['default']) * symbol_info.trade_tick_size
        logging.debug(f"Threshold: {threshold}")

        if threshold == 0:
            logging.error("Division by zero: threshold value is zero in check_entry_conditions")
            return False, False

        buy_condition &= close_price > max(wavy_c, wavy_h, wavy_l) + threshold
        sell_condition &= close_price < min(wavy_c, wavy_h, wavy_l) - threshold

    logging.debug(f"Final Buy condition: {buy_condition}")
    logging.debug(f"Final Sell condition: {sell_condition}")

    return buy_condition, sell_condition

def execute_trade(trade_request):
    logging.debug(f"Executing trade with request: {trade_request}")
    try:
        # Set initial trade details
        trade_request['entry_time'] = pd.Timestamp.now()
        trade_request['entry_price'] = trade_request['price']
        trade_request['profit'] = 0  # Initialize profit to 0

        # Place the order (simulated for this example)
        result = place_order(
            trade_request['symbol'],
            trade_request['action'].lower(),
            trade_request['volume'],
            trade_request['entry_price'],
            trade_request['sl'],
            trade_request['tp']
        )

        if result == 'Order failed':
            raise Exception("Failed to execute trade")

        logging.debug(f"Trade executed successfully: {result}")
        return result
    except Exception as e:
        handle_error(e, "Failed to execute trade")
        return None

def manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    logging.debug(f"Managing position for symbol: {symbol}")
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for position in positions:
                logging.debug(f"Checking position {position.ticket} with profit {position.profit}")

                current_equity = mt5.account_info().equity

                # Check for take profit, stop loss, or maximum loss per day conditions
                if position.profit >= min_take_profit:
                    logging.debug(f"Closing position {position.ticket} for profit")
                    close_position(position.ticket)
                    position['exit_time'] = pd.Timestamp.now()
                    position['exit_price'] = mt5.symbol_info_tick(symbol).bid
                    position['profit'] = (position['exit_price'] - position['entry_price']) * position['volume']

                elif position.profit <= -max_loss_per_day:
                    logging.debug(f"Closing position {position.ticket} for loss")
                    close_position(position.ticket)
                    position['exit_time'] = pd.Timestamp.now()
                    position['exit_price'] = mt5.symbol_info_tick(symbol).bid
                    position['profit'] = (position['exit_price'] - position['entry_price']) * position['volume']

                elif current_equity <= starting_equity * 0.9:  # Close position if equity drops by 10%
                    logging.debug(f"Closing position {position.ticket} due to equity drop")
                    close_position(position.ticket)
                    position['exit_time'] = pd.Timestamp.now()
                    position['exit_price'] = mt5.symbol_info_tick(symbol).bid
                    position['profit'] = (position['exit_price'] - position['entry_price']) * position['volume']

                elif mt5.positions_total() >= max_trades_per_day:
                    logging.debug(f"Closing position {position.ticket} due to max trades exceeded")
                    close_position(position.ticket)
                    position['exit_time'] = pd.Timestamp.now()
                    position['exit_price'] = mt5.symbol_info_tick(symbol).bid
                    position['profit'] = (position['exit_price'] - position['entry_price']) * position['volume']
    except Exception as e:
        handle_error(e, "Failed to manage position")

def calculate_tunnel_bounds(data, period, deviation_factor):
    data.loc[:, 'close'] = pd.to_numeric(data['close'], errors='coerce')  # Modified line
    logging.debug(f"Calculating tunnel bounds with period: {period} and deviation_factor: {deviation_factor}")

    if len(data) < period:
        return pd.Series([np.nan] * len(data)), pd.Series([np.nan] * len(data))

    ema = calculate_ema(data['close'], period)
    rolling_std = data['close'].rolling(window=period).std()

    volatility = rolling_std.mean()
    deviation = deviation_factor * volatility

    upper_bound = ema + deviation
    lower_bound = ema - deviation

    logging.debug(f"EMA Values for Tunnel Bounds: {ema}")
    logging.debug(f"Rolling Std: {rolling_std}")
    logging.debug(f"Volatility: {volatility}")
    logging.debug(f"Deviation: {deviation}")
    logging.debug(f"Upper Bound: {upper_bound}")
    logging.debug(f"Lower Bound: {lower_bound}")

    return upper_bound, lower_bound

def calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, pip_value):
    risk_amount = account_balance * risk_per_trade
    if stop_loss_pips == 0 or pip_value == 0:
        logging.error("Division by zero: stop_loss_pips or pip_value is zero in calculate_position_size")
        raise ZeroDivisionError("stop_loss_pips or pip_value cannot be zero")
    position_size = risk_amount / (stop_loss_pips * pip_value)
    logging.debug(f"Calculated position size: {position_size}")
    return position_size

def generate_trade_signal(data, period, deviation_factor):
    if len(data) < period:
        return None, None

    upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation_factor)
    
    last_close = pd.to_numeric(data['close'].iloc[-1], errors='coerce')
    upper_bound_last_value = upper_bound.iloc[-1]
    lower_bound_last_value = lower_bound.iloc[-1]

    logging.debug(f"Data: {data}")
    logging.debug(f"Upper Bound: {upper_bound}")
    logging.debug(f"Lower Bound: {lower_bound}")
    logging.debug(f"Last Close: {last_close}")
    logging.debug(f"Upper Bound Last Value: {upper_bound_last_value}")
    logging.debug(f"Lower Bound Last Value: {lower_bound_last_value}")

    if pd.isna(last_close) or pd.isna(upper_bound_last_value) or pd.isna(lower_bound_last_value):
        return None, None

    buy_condition = last_close >= upper_bound_last_value
    sell_condition = last_close <= lower_bound_last_value

    logging.debug(f"Buy Condition: {buy_condition}")
    logging.debug(f"Sell Condition: {sell_condition}")

    return buy_condition, sell_condition

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
            if data is None or data.empty:
                raise ValueError(f"Failed to retrieve historical data for {symbol}")

            period = 20
            market_conditions = 'volatile'  # Placeholder for determining market conditions
            deviation_factor = adjust_deviation_factor(market_conditions)

            logging.info("Calculating Wavy Tunnel indicators...")
            data['wavy_h'] = calculate_ema(data['high'], 34)
            data['wavy_c'] = calculate_ema(data['close'], 34)
            data['wavy_l'] = calculate_ema(data['low'], 34)
            data['tunnel1'] = calculate_ema(data['close'], 144)
            data['tunnel2'] = calculate_ema(data['close'], 169)
            data['long_term_ema'] = calculate_ema(data['close'], 200)
            logging.info("Indicators calculated.")

            logging.info("Detecting peaks and dips...")
            peak_type = 21  # Define the peak_type variable
            peaks, dips = detect_peaks_and_dips(data, peak_type)
            logging.info(f"Peaks: {peaks[:5]}")
            logging.info(f"Dips: {dips[:5]}")

            logging.info(f"Historical data shape after calculations: {data.shape}")
            logging.info(f"Historical data head after calculations:\n{data.head()}")

            logging.info("Generating entry signals...")
            data['buy_signal'], data['sell_signal'] = zip(*data.apply(lambda x: check_entry_conditions(x, peaks, dips, symbol), axis=1))
            logging.info("Entry signals generated.")

            if run_backtest:
                logging.info("Running backtest...")
                backtest_result = run_backtest(
                    symbol=symbol,
                    data=data,
                    initial_balance=starting_equity,
                    risk_percent=0.01,
                    min_take_profit=min_take_profit,
                    max_loss_per_day=max_loss_per_day,
                    starting_equity=starting_equity,
                    stop_loss_pips=20,
                    pip_value=0.0001,
                    max_trades_per_day=max_trades_per_day,
                    slippage=0,
                    transaction_cost=0
                )
                logging.info(f"Backtest result: {backtest_result}")
            else:
                buy_condition, sell_condition = generate_trade_signal(data, period, deviation_factor)

                if buy_condition:
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
                    logging.info(f"Executing BUY trade for {symbol}...")
                    execute_trade(trade_request)
                elif sell_condition:
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
                    logging.info(f"Executing SELL trade for {symbol}...")
                    execute_trade(trade_request)

                manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

    except Exception as e:
        handle_error(e, "Failed to run the strategy")
        raise
