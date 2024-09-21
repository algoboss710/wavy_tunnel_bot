import pandas as pd
import numpy as np
import logging
import MetaTrader5 as mt5
from datetime import datetime, time as dtime
from utils.error_handling import handle_error
import time
from config import Config
from metatrader.data_retrieval import get_historical_data


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
    logging.debug(f"Calculating EMA for period {period}")
    logging.debug(f"First few prices: {prices.head() if isinstance(prices, pd.Series) else prices[:5]}")

    prices = pd.Series(prices)
    prices = pd.to_numeric(prices, errors='coerce')
    ema_values = np.full(len(prices), np.nan, dtype=np.float64)
    if len(prices) < period:
        logging.warning(f"Not enough data for EMA calculation. Required: {period}, Available: {len(prices)}")
        return pd.Series(ema_values, index=prices.index)

    sma = np.mean(prices[:period])
    ema_values[period - 1] = sma
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema_values[i] = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]

    result = pd.Series(ema_values, index=prices.index)
    logging.debug(f"Calculated EMA. Last few values: {result.tail()}")
    return result

def detect_peaks_and_dips(df, peak_type=5):
    logging.debug(f"Detecting peaks and dips with peak_type: {peak_type}")
    highs = df['high'].values
    lows = df['low'].values
    center_index = peak_type // 2
    peaks = []
    dips = []

    for i in range(center_index, len(highs) - center_index):
        peak_window = highs[i - center_index:i + center_index + 1]
        dip_window = lows[i - center_index:i + center_index + 1]

        logging.debug(f"Analyzing peak window: {peak_window}")
        logging.debug(f"Analyzing dip window: {dip_window}")

        if all(peak_window[center_index] > peak_window[j] for j in range(len(peak_window)) if j != center_index):
            peaks.append(highs[i])
            logging.debug(f"Peak detected at index {i}: {highs[i]}")

        if all(dip_window[center_index] < dip_window[j] for j in range(len(dip_window)) if j != center_index):
            dips.append(lows[i])
            logging.debug(f"Dip detected at index {i}: {lows[i]}")

    logging.info(f"Total peaks detected: {len(peaks)}, Peaks: {peaks}")
    logging.info(f"Total dips detected: {len(dips)}, Dips: {dips}")
    return peaks, dips

def check_entry_conditions(row, peaks, dips, symbol):
    wavy_c, wavy_h, wavy_l = row['wavy_c'], row['wavy_h'], row['wavy_l']
    tunnel1, tunnel2 = row['tunnel1'], row['tunnel2']
    close_price = row['close']

    logging.info(f"Checking entry conditions for {symbol}:")
    logging.info(f"Close price: {close_price:.5f}")
    logging.info(f"Wavy C: {wavy_c:.5f}, Wavy H: {wavy_h:.5f}, Wavy L: {wavy_l:.5f}")
    logging.info(f"Tunnel1: {tunnel1:.5f}, Tunnel2: {tunnel2:.5f}")

    logging.debug(f"Number of peaks detected: {len(peaks)}, Peaks (first 5): {peaks[:5]}")
    logging.debug(f"Number of dips detected: {len(dips)}, Dips (first 5): {dips[:5]}")

    buy_condition1 = close_price > max(wavy_c, wavy_h, wavy_l)
    buy_condition2 = min(wavy_c, wavy_h, wavy_l) > max(tunnel1, tunnel2)
    buy_condition3 = any(abs(close_price - peak) <= 0.001 for peak in peaks)
    logging.debug(f"Buy Condition 1 (close > max(Wavy C, Wavy H, Wavy L)): {buy_condition1}")
    logging.debug(f"Buy Condition 2 (min(Wavy C, Wavy H, Wavy L) > max(Tunnel1, Tunnel2)): {buy_condition2}")
    logging.debug(f"Buy Condition 3 (close price near peak): {buy_condition3}")
    sell_condition1 = close_price < min(wavy_c, wavy_h, wavy_l)
    sell_condition2 = max(wavy_c, wavy_h, wavy_l) < min(tunnel1, tunnel2)
    sell_condition3 = any(abs(close_price - dip) <= 0.001 for dip in dips)
    logging.debug(f"Sell Condition 1 (close < min(Wavy C, Wavy H, Wavy L)): {sell_condition1}")
    logging.debug(f"Sell Condition 2 (max(Wavy C, Wavy H, Wavy L) < min(Tunnel1, Tunnel2)): {sell_condition2}")
    logging.debug(f"Sell Condition 3 (close price near dip): {sell_condition3}")

    buy_condition = buy_condition1 or buy_condition2 or buy_condition3
    sell_condition = sell_condition1 or sell_condition2 or sell_condition3

  # Log the final conditions
    logging.info(f"Buy conditions: {buy_condition1}, {buy_condition2}, {buy_condition3}")
    logging.info(f"Sell conditions: {sell_condition1}, {sell_condition2}, {sell_condition3}")
    logging.info(f"Buy condition met: {buy_condition}")
    logging.info(f"Sell condition met: {sell_condition}")

    buy_reasons = [f"Condition {i+1}: {cond}" for i, cond in enumerate([buy_condition1, buy_condition2, buy_condition3]) if cond]
    sell_reasons = [f"Condition {i+1}: {cond}" for i, cond in enumerate([sell_condition1, sell_condition2, sell_condition3]) if cond]

    logging.info(f"Entry conditions for {symbol}: Buy = {buy_condition} (reasons: {', '.join(buy_reasons)}), Sell = {sell_condition} (reasons: {', '.join(sell_reasons)})")

    return buy_condition, sell_condition


def execute_trade(trade_request, retries=4, delay=6):
    attempt = 0
    while attempt <= retries:
        try:
            logging.info(f"Attempting to execute trade: {trade_request}")

            if not ensure_symbol_subscription(trade_request['symbol']):
                logging.error(f"Failed to subscribe to symbol {trade_request['symbol']}")
                return None

            if not check_broker_connection() or not check_market_open():
                logging.error("Trade execution aborted due to connection issues or market being closed.")
                return None

            modified_request = trade_request.copy()
            modified_request['action'] = mt5.TRADE_ACTION_DEAL

            # Log the SL and TP values before sending the order
            logging.info(f"Setting SL: {modified_request['sl']}, TP: {modified_request['tp']} before sending order.")
            logging.info(f"Placing order with price: {modified_request['price']} and volume: {modified_request['volume']}")

            result = mt5.order_send(modified_request)

            if result is None:
                error_code = mt5.last_error()
                logging.error(f"Failed to place order: mt5.order_send returned None. Error code: {error_code}")
                raise ValueError(f"mt5.order_send returned None. Error code: {error_code}")

            logging.info(f"Order response received at {datetime.now()}: {result}")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Trade executed successfully: {result}")
                return result
            elif result.retcode == 10009:
                logging.warning("Order placed successfully, but not yet executed")
                return result
            elif result.retcode == 10004:  # Requote
                logging.warning("Requote error. Retrying with updated price.")
                current_price = mt5.symbol_info_tick(trade_request['symbol']).ask if trade_request['type'] == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(trade_request['symbol']).bid
                modified_request['price'] = current_price
                continue
            elif result.retcode == 10016:  # Invalid stops
                logging.error("Invalid stops. Adjusting SL and TP.")
                symbol_info = mt5.symbol_info(trade_request['symbol'])
                min_stop_level = symbol_info.trade_stops_level * symbol_info.point
                if trade_request['type'] == mt5.ORDER_TYPE_BUY:
                    modified_request['sl'] = min(modified_request['sl'], modified_request['price'] - min_stop_level)
                    modified_request['tp'] = max(modified_request['tp'], modified_request['price'] + min_stop_level)
                else:
                    modified_request['sl'] = max(modified_request['sl'], modified_request['price'] + min_stop_level)
                    modified_request['tp'] = min(modified_request['tp'], modified_request['price'] - min_stop_level)
                continue
            else:
                logging.error(f"Order failed with retcode: {result.retcode}, Comment: {result.comment}")
                raise ValueError(f"Order failed: {result.comment}")

        except Exception as e:
            logging.error(f"Exception occurred during trade execution attempt {attempt + 1}: {str(e)}")
            attempt += 1

        if attempt <= retries:
            logging.info(f"Retrying in {delay} seconds... Current attempt: {attempt}/{retries}")
            time.sleep(delay)

    logging.error("Trade execution failed after maximum retries.")
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
            logging.error(f"Failed to place pending order for {trade_request['symbol']}: {result.comment}")
            return None
        logging.info(f"Pending order placed successfully for {trade_request['symbol']}")
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
                logging.info(f"Managing position for {symbol}. Current profit: {position.profit}, Equity: {current_equity}")

                if position.profit >= min_take_profit:
                    logging.info(f"Profit target reached for {symbol}. Closing position.")
                    close_result = close_position(position.ticket)
                    logging.info(f"Close position result: {close_result}")

                elif position.profit <= -max_loss_per_day:
                    logging.info(f"Loss limit reached for {symbol}. Closing position.")
                    close_result = close_position(position.ticket)
                    logging.info(f"Close position result: {close_result}")

                elif current_equity <= starting_equity * 0.9:
                    logging.info(f"Drawdown limit reached for {symbol}. Closing position.")
                    close_result = close_position(position.ticket)
                    logging.info(f"Close position result: {close_result}")

                elif mt5.positions_total() >= max_trades_per_day:
                    logging.info(f"Trade limit reached for the day. Closing position for {symbol}.")
                    close_result = close_position(position.ticket)
                    logging.info(f"Close position result: {close_result}")

                # Log the profit and details of the closed position
                position['exit_time'] = pd.Timestamp.now()
                position['exit_price'] = mt5.symbol_info_tick(symbol).bid
                position['profit'] = (position['exit_price'] - position['entry_price']) * position['volume']
                logging.info(f"Position details - Exit Time: {position['exit_time']}, Exit Price: {position['exit_price']}, Profit: {position['profit']}")

        else:
            logging.info(f"No open positions found for {symbol}.")
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

    position_size_base = risk_amount / (stop_loss_pips * pip_value)
    position_size_lots = position_size_base / 100000  # Convert to lots

    position_size_lots = min(position_size_lots, 0.1)
    position_size_lots = max(position_size_lots, 0.01)

    logging.info(f"Calculated position size: {position_size_lots} lots")

    return round(position_size_lots, 2)

def generate_trade_signal(data, period, deviation_factor):
    if len(data) < period:
        logging.warning(f"Not enough data to generate trade signal. Required: {period}, Available: {len(data)}")
        return None, None

    upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation_factor)

    last_close = pd.to_numeric(data['close'].iloc[-1], errors='coerce')
    upper_bound_last_value = upper_bound.iloc[-1]
    lower_bound_last_value = lower_bound.iloc[-1]

    logging.info(f"Generating trade signal with close price: {last_close}, upper bound: {upper_bound_last_value}, lower bound: {lower_bound_last_value}")

    if pd.isna(last_close) or pd.isna(upper_bound_last_value) or pd.isna(lower_bound_last_value):
        logging.error("One or more values are NaN, cannot generate trade signal.")
        return None, None

    buy_condition = last_close >= upper_bound_last_value
    sell_condition = last_close <= lower_bound_last_value

    logging.info(f"Buy condition: {buy_condition}, Sell condition: {sell_condition}")

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
def run_strategy(symbols, mt5_init, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day, run_backtest, data=None, std_dev=None):
    try:
        logging.info("Starting strategy execution")
        total_profit = 0
        total_loss = 0
        max_drawdown = 0
        current_balance = starting_equity
        peak_balance = starting_equity

        for symbol in symbols:
            logging.info(f"Processing symbol: {symbol}")
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
            logging.info(f"Detected peaks: {peaks[:5]} (total: {len(peaks)}), dips: {dips[:5]} (total: {len(dips)})")

            logging.info("Generating entry signals...")
            data['buy_signal'], data['sell_signal'] = zip(*data.apply(lambda x: check_entry_conditions(x, peaks, dips, symbol), axis=1))

            buy_condition, sell_condition = generate_trade_signal(data, period, deviation_factor)

            logging.info(f"Buy Condition: {buy_condition}, Sell Condition: {sell_condition}")

            if buy_condition or sell_condition:
                current_tick = get_current_data(symbol)
                logging.info(f"Latest price data for {symbol}: {current_tick}")

                trade_request = {
                    'action': mt5.TRADE_ACTION_DEAL,
                    'symbol': symbol,
                    'volume': lot_size,
                    'type': mt5.ORDER_TYPE_BUY if buy_condition else mt5.ORDER_TYPE_SELL,
                    'price': current_tick['bid'] if buy_condition else current_tick['ask'],
                    'sl': current_tick['bid'] - (1.5 * std_dev) if buy_condition else current_tick['ask'] + (1.5 * std_dev),
                    'tp': current_tick['bid'] + (2 * std_dev) if buy_condition else current_tick['ask'] - (2 * std_dev),
                    'deviation': 10,
                    'magic': 12345,
                    'comment': 'Tunnel Strategy',
                    'type_filling': mt5.ORDER_FILLING_FOK,
                    'type_time': mt5.ORDER_TIME_GTC
                }

                logging.info(f"Trade request: {trade_request}")
                result = execute_trade(trade_request)
                if result:
                    profit = trade_request['tp'] - trade_request['price'] if buy_condition else trade_request['price'] - trade_request['tp']
                    total_profit += profit
                    current_balance += profit
                    peak_balance = max(peak_balance, current_balance)
                    drawdown = peak_balance - current_balance
                    max_drawdown = max(max_drawdown, drawdown)
                    logging.info(f"Trade executed successfully. Profit: {profit:.2f}, Current Balance: {current_balance:.2f}, Drawdown: {drawdown:.2f}")
                else:
                    logging.error("Trade execution failed")

            manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

        logging.info("Strategy execution completed")
        return {
            'total_profit': total_profit,
            'total_loss': total_loss,
            'max_drawdown': max_drawdown
        }

    except Exception as e:
        handle_error(e, "Failed to run the strategy")
        return None

def get_data(symbol, mode='live', start_date=None, end_date=None, timeframe=mt5.TIMEFRAME_H1):
    """
    Retrieve data based on the mode. Can be used for both live and historical data.
    - mode: "live" or "backtest"
    - If backtest mode, start_date and end_date must be provided.
    """
    if mode == 'backtest':
        # Fetch historical data for backtesting
        return get_historical_data(symbol, timeframe, start_date, end_date)  # Directly call get_historical_data from metatrader
    else:
        # Fetch live tick data
        current_tick = get_current_data(symbol)
        if current_tick:
            live_data = pd.DataFrame([{
                'time': current_tick['time'],
                'open': current_tick['last'] if current_tick['last'] != 0 else (current_tick['bid'] + current_tick['ask']) / 2,
                'high': current_tick['ask'],
                'low': current_tick['bid'],
                'close': current_tick['last'] if current_tick['last'] != 0 else (current_tick['bid'] + current_tick['ask']) / 2,
                'volume': current_tick['volume'],
                'bid': current_tick['bid'],
                'ask': current_tick['ask']
            }])
            logging.info(f"Fetched live data for {symbol}: {live_data.tail()}")
            return live_data
        else:
            logging.error(f"Failed to retrieve live data for {symbol}")
            return None


def place_order(symbol, action, volume, price, sl, tp):
    try:
        logging.info(f"Preparing to place order for {symbol} - Action: {action}, Volume: {volume}, Price: {price}, SL: {sl}, TP: {tp}")
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
        logging.info(f"Order send result for {symbol}: {result}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to place order for {symbol}: {result.comment}")
            return 'Order failed'
        logging.info(f"Order placed successfully for {symbol}")
        return 'Order placed'
    except Exception as e:
        logging.error(f"Exception occurred while placing order for {symbol}: {str(e)}")
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
    logging.info(f"Attempting to retrieve tick data for symbol: {symbol}")
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        tick_data = {
            'symbol': symbol,
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume
        }
        logging.info(f"Retrieved tick data for {symbol}: {tick_data}")
        return tick_data
    else:
        logging.error(f"Failed to retrieve fresh tick data for {symbol}")
        raise ValueError(f"Failed to retrieve fresh tick data for {symbol}")