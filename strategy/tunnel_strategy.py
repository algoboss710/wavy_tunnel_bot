import pandas as pd
import numpy as np
import logging
import MetaTrader5 as mt5
from datetime import datetime, time as dtime
from utils.error_handling import handle_error
import time
from config import Config

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_current_data(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        tick_data = {
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'spread': tick.ask - tick.bid,
            'volume': tick.volume
        }
        logging.info(f"Retrieved tick data for {symbol}: {tick_data}")
        return tick_data
    else:
        raise ValueError(f"Failed to retrieve current tick data for {symbol}")

def calculate_ema(prices, period):
    prices = pd.Series(prices)
    prices = pd.to_numeric(prices, errors='coerce')
    ema_values = np.full(len(prices), np.nan, dtype=np.float64)
    if len(prices) < period:
        return pd.Series(ema_values, index=prices.index)

    sma = np.mean(prices[:period])
    ema_values[period - 1] = sma
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]

    return pd.Series(ema_values, index=prices.index)

def detect_peaks_and_dips(df, peak_type):
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

    return peaks, dips

def check_entry_conditions(row, peaks, dips, symbol):
    wavy_c, wavy_h, wavy_l = row['wavy_c'], row['wavy_h'], row['wavy_l']
    tunnel1, tunnel2 = row['tunnel1'], row['tunnel2']
    close_price = row['close']

    buy_condition = (
        close_price > max(wavy_c, wavy_h, wavy_l) and
        min(wavy_c, wavy_h, wavy_l) > max(tunnel1, tunnel2) and
        any(abs(close_price - peak) <= 0.001 for peak in peaks)
    )
    sell_condition = (
        close_price < min(wavy_c, wavy_h, wavy_l) and
        max(wavy_c, wavy_h, wavy_l) < min(tunnel1, tunnel2) and
        any(abs(close_price - dip) <= 0.001 for dip in dips)
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
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logging.error(f"Failed to get symbol info for {symbol}")
            return False, False

        threshold = threshold_values.get(symbol[:3], threshold_values['default']) * symbol_info.trade_tick_size

        if threshold == 0:
            logging.error("Division by zero: threshold value is zero in check_entry_conditions")
            return False, False

        buy_condition &= close_price > max(wavy_c, wavy_h, wavy_l) + threshold
        sell_condition &= close_price < min(wavy_c, wavy_h, wavy_l) - threshold

    logging.debug(f"Primary entry conditions for {symbol} at {row['time']}: buy={buy_condition}, sell={sell_condition}")
    return buy_condition, sell_condition

def execute_trade(trade_request, retries=4, delay=6, is_backtest=False):
    if is_backtest:
        # For backtesting, we'll just log the trade and return a success result
        logging.info(f"Backtest: Executing trade - {trade_request}")
        return True

    attempt = 0
    while attempt <= retries:
        try:
            logging.debug(f"Attempt {attempt + 1} to execute trade with request: {trade_request}")
            if not ensure_symbol_subscription(trade_request['symbol']):
                logging.error(f"Failed to subscribe to symbol {trade_request['symbol']}")
                return None
            if not check_broker_connection() or not check_market_open():
                logging.error("Trade execution aborted due to connection issues or market being closed.")
                return None
            logging.info(f"Placing order with price: {trade_request['price']}")
            result = mt5.order_send(trade_request)
            if result is None:
                logging.error(f"Failed to place order: mt5.order_send returned None. Trade Request: {trade_request}")
                raise ValueError("mt5.order_send returned None.")
            logging.info(f"Order response received at {datetime.now()}: {result}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Trade executed successfully: {result}")
                return result
            elif result.retcode == 10021:
                logging.warning(f"Failed to execute trade due to 'No prices' error. Attempt {attempt + 1} of {retries + 1}")
                attempt += 1
                if attempt <= retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(f"Failed to execute trade after {retries + 1} attempts due to 'No prices' error.")
            else:
                logging.error(f"Failed to execute trade: {result.retcode} - {result.comment}")
                return None
        except Exception as e:
            logging.error(f"Exception occurred during trade execution attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt <= retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Failed to execute trade after {retries + 1} attempts due to an exception.")
                return None
    return None

def place_pending_order(trade_request):
    try:
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if trade_request['action'] == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_SELL_LIMIT
        adjusted_sl = trade_request['sl'] - Config.SL_TP_ADJUSTMENT_PIPS if trade_request['action'] == mt5.ORDER_TYPE_BUY else trade_request['sl'] + Config.SL_TP_ADJUSTMENT_PIPS
        adjusted_tp = trade_request['tp'] + Config.SL_TP_ADJUSTMENT_PIPS if trade_request['action'] == mt5.ORDER_TYPE_BUY else trade_request['tp'] - Config.SL_TP_ADJUSTMENT_PIPS

        pending_order_request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": trade_request['symbol'],
            "volume": trade_request['volume'],
            "type": order_type,
            "price": trade_request['price'],
            "sl": adjusted_sl,
            "tp": adjusted_tp,
            "deviation": trade_request['deviation'],
            "magic": trade_request['magic'],
            "comment": f"Pending {trade_request['comment']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        logging.debug(f"Placing pending order: {pending_order_request}")
        result = mt5.order_send(pending_order_request)
        logging.info(f"Pending order result: {result}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to place pending order: {result.comment}")
            return None
        return result
    except Exception as e:
        handle_error(e, "Failed to place pending order")
        return None

def manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for position in positions:
                current_equity = mt5.account_info().equity

                if position.profit >= min_take_profit:
                    close_position(position.ticket)
                    position['exit_time'] = pd.Timestamp.now()
                    position['exit_price'] = mt5.symbol_info_tick(symbol).bid
                    position['profit'] = (position['exit_price'] - position['entry_price']) * position['volume']

                elif position.profit <= -max_loss_per_day:
                    close_position(position.ticket)
                    position['exit_time'] = pd.Timestamp.now()
                    position['exit_price'] = mt5.symbol_info_tick(symbol).bid
                    position['profit'] = (position['exit_price'] - position['entry_price']) * position['volume']

                elif current_equity <= starting_equity * 0.9:
                    close_position(position.ticket)
                    position['exit_time'] = pd.Timestamp.now()
                    position['exit_price'] = mt5.symbol_info_tick(symbol).bid
                    position['profit'] = (position['exit_price'] - position['entry_price']) * position['volume']

                elif mt5.positions_total() >= max_trades_per_day:
                    close_position(position.ticket)
                    position['exit_time'] = pd.Timestamp.now()
                    position['exit_price'] = mt5.symbol_info_tick(symbol).bid
                    position['profit'] = (position['exit_price'] - position['entry_price']) * position['volume']
    except Exception as e:
        handle_error(e, "Failed to manage position")

def calculate_tunnel_bounds(data, period, deviation_factor):
    if len(data) < period:
        return pd.Series([np.nan] * len(data)), pd.Series([np.nan] * len(data))

    data = data.copy()
    data['close'] = pd.to_numeric(data['close'], errors='coerce')

    ema = calculate_ema(data['close'], period)
    rolling_std = data['close'].rolling(window=period).std()
    volatility = rolling_std * deviation_factor
    deviation = volatility / np.sqrt(period)
    upper_bound = ema + deviation
    lower_bound = ema - deviation

    return upper_bound, lower_bound

def calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, pip_value):
    risk_amount = account_balance * risk_per_trade
    if stop_loss_pips == 0 or pip_value == 0:
        logging.error("Division by zero: stop_loss_pips or pip_value is zero in calculate_position_size")
        raise ZeroDivisionError("stop_loss_pips or pip_value cannot be zero")
    position_size = risk_amount / (stop_loss_pips * pip_value)

    logging.debug(f"Calculated position size: {position_size}. Inputs: account_balance={account_balance}, risk_per_trade={risk_per_trade}, stop_loss_pips={stop_loss_pips}, pip_value={pip_value}")
    return position_size

def generate_trade_signal(data, period, deviation_factor):
    if len(data) < period:
        return None, None

    upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation_factor)

    last_close = pd.to_numeric(data['close'].iloc[-1], errors='coerce')
    upper_bound_last_value = upper_bound.iloc[-1]
    lower_bound_last_value = lower_bound.iloc[-1]

    if pd.isna(last_close) or pd.isna(upper_bound_last_value) or pd.isna(lower_bound_last_value):
        return None, None

    buy_condition = last_close >= upper_bound_last_value
    sell_condition = last_close <= lower_bound_last_value

    return buy_condition, sell_condition

def adjust_deviation_factor(market_conditions):
    if market_conditions == 'volatile':
        return 2.5
    else:
        return 2.0

def ensure_symbol_subscription(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logging.error(f"Symbol {symbol} is not available.")
        return False

    if not symbol_info.visible:
        logging.info(f"Symbol {symbol} is not visible, attempting to make it visible.")
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to select symbol {symbol}")
            return False

    logging.info(f"Symbol {symbol} is already subscribed and visible.")
    return True

def detect_crossover(array1, array2):
    """Detect if array1 has crossed over array2."""
    if len(array1) < 2 or len(array2) < 2:
        logging.warning("Not enough data points to detect crossover")
        return False
    return (array1[-2] <= array2[-2]) and (array1[-1] > array2[-1])

def detect_crossunder(array1, array2):
    """Detect if array1 has crossed under array2."""
    if len(array1) < 2 or len(array2) < 2:
        logging.warning("Not enough data points to detect crossunder")
        return False
    return (array1[-2] >= array2[-2]) and (array1[-1] < array2[-1])

def check_secondary_entry_conditions(data, symbol):
    logging.debug(f"Checking secondary entry conditions for {symbol}")
    
    try:
        wavy_c, wavy_h, wavy_l = data['wavy_c'].values, data['wavy_h'].values, data['wavy_l'].values
        tunnel1, tunnel2 = data['tunnel1'].values, data['tunnel2'].values
        close = data['close'].values

        max_wavy = np.maximum.reduce([wavy_c, wavy_h, wavy_l])
        min_wavy = np.minimum.reduce([wavy_c, wavy_h, wavy_l])
        min_tunnel = np.minimum(tunnel1, tunnel2)
        max_tunnel = np.maximum(tunnel1, tunnel2)

        logging.debug(f"Last close price: {close[-1]}")
        logging.debug(f"Max wavy: {max_wavy[-1]}, Min wavy: {min_wavy[-1]}")
        logging.debug(f"Min tunnel: {min_tunnel[-1]}, Max tunnel: {max_tunnel[-1]}")

        secondary_long_condition = detect_crossover(close, max_wavy) and (close[-1] < min_tunnel[-1])
        secondary_short_condition = detect_crossunder(close, min_wavy) and (close[-1] > max_tunnel[-1])

        logging.debug(f"Initial conditions - Long: {secondary_long_condition}, Short: {secondary_short_condition}")

        # Implement minimum gap check
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Failed to get symbol info for {symbol}")
        
        min_gap = Config.MIN_GAP_SECOND_VALUE * symbol_info.point
        price_diff_long = min_tunnel[-1] - close[-1]
        price_diff_short = close[-1] - max_tunnel[-1]

        logging.debug(f"Minimum gap: {min_gap}")
        logging.debug(f"Price diff long: {price_diff_long}, Price diff short: {price_diff_short}")

        if secondary_long_condition:
            secondary_long_condition &= price_diff_long >= min_gap
            logging.debug(f"Long condition after min gap check: {secondary_long_condition}")

        if secondary_short_condition:
            secondary_short_condition &= price_diff_short >= min_gap
            logging.debug(f"Short condition after min gap check: {secondary_short_condition}")

        # Implement zone limitation check
        if secondary_long_condition:
            zone_percentage = (close[-1] - max_wavy[-1]) / (min_tunnel[-1] - max_wavy[-1])
            secondary_long_condition &= zone_percentage <= Config.MAX_ALLOW_INTO_ZONE
            logging.debug(f"Long zone percentage: {zone_percentage}, Condition after check: {secondary_long_condition}")

        if secondary_short_condition:
            zone_percentage = (min_wavy[-1] - close[-1]) / (min_wavy[-1] - max_tunnel[-1])
            secondary_short_condition &= zone_percentage <= Config.MAX_ALLOW_INTO_ZONE
            logging.debug(f"Short zone percentage: {zone_percentage}, Condition after check: {secondary_short_condition}")

        logging.info(f"Final secondary entry conditions for {symbol}: long={secondary_long_condition}, short={secondary_short_condition}")
        return secondary_long_condition, secondary_short_condition

    except Exception as e:
        logging.error(f"Error in check_secondary_entry_conditions for {symbol}: {str(e)}")
        return False, False
def run_strategy(symbols, mt5_init, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day, run_backtest, data=None, std_dev=None):
    try:
        total_profit = 0
        total_loss = 0
        max_drawdown = 0
        current_balance = starting_equity
        peak_balance = starting_equity

        for symbol in symbols:
            if data is None:
                current_data = get_current_data(symbol)
                logging.info(f"Current data for {symbol}: {current_data}")

                data = pd.DataFrame([{
                    'time': current_data['time'],
                    'open': current_data['last'],
                    'high': current_data['last'],
                    'low': current_data['last'],
                    'close': current_data['last'],
                    'volume': 0
                }])
            else:
                logging.info(f"Using provided data for {symbol}")

            period = 20
            market_conditions = 'volatile'
            deviation_factor = adjust_deviation_factor(market_conditions)

            logging.info("Calculating Wavy Tunnel indicators...")
            data['wavy_h'] = calculate_ema(data['high'], 34)
            data['wavy_c'] = calculate_ema(data['close'], 34)
            data['wavy_l'] = calculate_ema(data['low'], 34)
            data['tunnel1'] = calculate_ema(data['close'], 144)
            data['tunnel2'] = calculate_ema(data['close'], 169)
            data['long_term_ema'] = calculate_ema(data['close'], 200)

            logging.info("Detecting peaks and dips...")
            peak_type = 21
            peaks, dips = detect_peaks_and_dips(data, peak_type)
            logging.info(f"Peaks: {peaks[:5]}")
            logging.info(f"Dips: {dips[:5]}")

            logging.info("Generating entry signals...")
            data['primary_buy'], data['primary_sell'] = zip(*data.apply(lambda x: check_entry_conditions(x, peaks, dips, symbol), axis=1))

            if Config.ENABLE_SECONDARY_STRATEGY:
                data['secondary_buy'], data['secondary_sell'] = zip(*data.apply(lambda x: check_secondary_entry_conditions(x, symbol), axis=1))

            for i in range(len(data)):
                row = data.iloc[i]
                if i >= 1:  # We need at least two rows for crossover/under detection
                    primary_buy, primary_sell = check_entry_conditions(row, peaks, dips, symbol)
                
                if Config.ENABLE_SECONDARY_STRATEGY:
                    secondary_buy, secondary_sell = check_secondary_entry_conditions(data.iloc[i-1:i+1], symbol)
                    if secondary_buy or secondary_sell:
                        secondary_strategy_triggers += 1
                        logging.info(f"Secondary strategy triggered for {symbol} at index {i}: Buy={secondary_buy}, Sell={secondary_sell}")
                else:
                    secondary_buy, secondary_sell = False, False

                buy_condition = primary_buy or (Config.ENABLE_SECONDARY_STRATEGY and secondary_buy)
                sell_condition = primary_sell or (Config.ENABLE_SECONDARY_STRATEGY and secondary_sell)

                if buy_condition or sell_condition:
                    logging.info(f"Trade signal for {symbol} at index {i}: Buy={buy_condition}, Sell={sell_condition}")
                    
                    current_tick = get_current_data(symbol) if data is None else row
                    std_dev = data['close'].rolling(window=20).std().iloc[i] if std_dev is None else std_dev

                    trade_request = {
                        'action': 'BUY' if buy_condition else 'SELL',
                        'symbol': symbol,
                        'volume': lot_size,
                        'price': current_tick['bid'] if buy_condition else current_tick['ask'],
                        'sl': current_tick['bid'] - (1.5 * std_dev) if buy_condition else current_tick['ask'] + (1.5 * std_dev),
                        'tp': current_tick['bid'] + (2 * std_dev) if buy_condition else current_tick['ask'] - (2 * std_dev),
                        'deviation': 10,
                        'magic': 12345,
                        'comment': 'Tunnel Strategy - Secondary' if (secondary_buy or secondary_sell) else 'Tunnel Strategy - Primary',
                        'type': 'ORDER_TYPE_BUY' if buy_condition else 'ORDER_TYPE_SELL',
                        'type_filling': 'ORDER_FILLING_FOK',
                        'type_time': 'ORDER_TIME_GTC'
                    }

                    logging.info(f"Attempting to execute trade: {trade_request}")
                    result = execute_trade(trade_request)

                    if result:
                        profit = trade_request['tp'] - trade_request['price'] if buy_condition else trade_request['price'] - trade_request['tp']
                        total_profit += profit
                        current_balance += profit
                        peak_balance = max(peak_balance, current_balance)
                        drawdown = peak_balance - current_balance
                        max_drawdown = max(max_drawdown, drawdown)
                        logging.info(f"Trade executed successfully. Profit: {profit}, Current Balance: {current_balance}")
                    else:
                        logging.error("Trade execution failed")

                manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

        logging.info("Strategy execution completed")
        logging.info(f"Total profit: {total_profit}")
        logging.info(f"Max drawdown: {max_drawdown}")
        logging.info(f"Total secondary strategy triggers: {secondary_strategy_triggers}")

        return {
            'total_profit': total_profit,
            'total_loss': total_loss,
            'max_drawdown': max_drawdown,
            'secondary_strategy_triggers': secondary_strategy_triggers
        }

    except Exception as e:
        handle_error(e, "Failed to run the strategy")
        return None

def place_order(symbol, action, volume, price, sl, tp):
    try:
        order_type = mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 12345,
            "comment": "Tunnel Strategy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        logging.debug(f"Placing order: {order}")
        result = mt5.order_send(order)
        logging.info(f"Order send result: {result}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to place order: {result.comment}")
            return 'Order failed'
        return 'Order placed'
    except Exception as e:
        logging.error(f"Failed to place order: {str(e)}")
        return 'Order failed'

def close_position(ticket):
    try:
        position = mt5.positions_get(ticket=ticket)
        if position:
            close_request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': position[0].symbol,
                'volume': position[0].volume,
                'type': mt5.ORDER_TYPE_SELL if position[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                'position': ticket,
                'price': mt5.symbol_info_tick(position[0].symbol).bid if position[0].type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position[0].symbol).ask,
                'deviation': 10,
                'magic': 12345,
                'comment': 'Tunnel Strategy Close',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_FOK,
            }

            logging.debug(f"Closing position with request: {close_request}")
            result = mt5.order_send(close_request)
            logging.info(f"Close position result: {result}")

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Failed to close position: {result.comment}")
                return 'Close failed'
            return 'Position closed'
        return 'Position not found'
    except Exception as e:
        logging.error(f"Failed to close position: {str(e)}")
        return 'Close failed'

def check_broker_connection():
    if not mt5.terminal_info().connected:
        logging.error("Broker is not connected.")
        return False
    logging.info("Broker is connected.")
    return True

def check_market_open():
    current_time = datetime.now().time()
    market_open = dtime(0, 0)
    market_close = dtime(23, 59)
    if not (market_open <= current_time <= market_close):
        logging.error("Market is closed.")
        return False
    logging.info("Market is open.")
    return True

def get_fresh_tick_data(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        return {
            'symbol': symbol,
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume
        }
    else:
        raise ValueError(f"Failed to retrieve fresh tick data for {symbol}")

