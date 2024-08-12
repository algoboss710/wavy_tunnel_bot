#Tunnel_strategy.py:

import pandas as pd
import numpy as np
import logging
import MetaTrader5 as mt5
from datetime import datetime, time as dtime
from utils.error_handling import handle_error
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_current_data(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        tick_data = {
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last
        }
        logging.info(f"Retrieved tick data for {symbol}: {tick_data}")
        return tick_data
    else:
        raise ValueError(f"Failed to retrieve current tick data for {symbol}")

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

    ema_series = pd.Series(ema_values, index=prices.index)
    return ema_series

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
        any(abs(close_price - peak) <= 0.001 for peak in peaks)
    )
    sell_condition = (
        close_price < min(wavy_c, wavy_h, wavy_l) and
        max(wavy_c, wavy_h, wavy_l) < min(tunnel1, tunnel2) and
        any(abs(close_price - dip) <= 0.001 for dip in dips)
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

def execute_trade(trade_request, retries=4, delay=4):
    attempt = 0
    last_tick_time = None
    while attempt <= retries:
        try:
            logging.debug(f"Attempt {attempt + 1} to execute trade with request: {trade_request}")

            latest_data = get_current_data(trade_request['symbol'])

            if last_tick_time and last_tick_time == latest_data['time']:
                logging.warning(f"Tick data for {trade_request['symbol']} has not been updated since last attempt.")
            else:
                last_tick_time = latest_data['time']

            logging.info(f"Latest price data for {trade_request['symbol']} at {datetime.now()}: {latest_data}")

            trade_request['price'] = latest_data['bid'] if trade_request['action'] == 'BUY' else latest_data['ask']
            trade_request['sl'] = trade_request['price'] - (1.5 * trade_request.get('std_dev', 0)) if trade_request['action'] == 'BUY' else trade_request['price'] + (1.5 * trade_request.get('std_dev', 0))
            trade_request['tp'] = trade_request['price'] + (2 * trade_request.get('std_dev', 0)) if trade_request['action'] == 'BUY' else trade_request['price'] - (2 * trade_request.get('std_dev', 0))

            logging.info(f"Placing order with updated price: {trade_request}")

            order = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': trade_request['symbol'],
                'volume': trade_request['volume'],
                'type': mt5.ORDER_TYPE_BUY if trade_request['action'] == 'BUY' else mt5.ORDER_TYPE_SELL,
                'price': trade_request['price'],
                'sl': trade_request['sl'],
                'tp': trade_request['tp'],
                'deviation': trade_request['deviation'],
                'magic': trade_request['magic'],
                'comment': trade_request['comment'],
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_FOK,
            }

            result = mt5.order_send(order)

            logging.info(f"Order response received at {datetime.now()}: {result}")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.debug(f"Trade executed successfully: {result}")
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
            handle_error(e, f"Exception occurred during trade execution attempt {attempt + 1}")
            attempt += 1
            if attempt <= retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Failed to execute trade after {retries + 1} attempts due to an exception.")
                return None

    return None

def manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    logging.debug(f"Managing position for symbol: {symbol}")
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for position in positions:
                logging.debug(f"Checking position {position.ticket} with profit {position.profit}")

                current_equity = mt5.account_info().equity

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

                elif current_equity <= starting_equity * 0.9:
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
    logging.debug(f"Calculating tunnel bounds with period: {period} and deviation_factor: {deviation_factor}")

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

    logging.debug(f"EMA: {ema}")
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
            logging.info("Indicators calculated.")

            logging.info("Detecting peaks and dips...")
            peak_type = 21
            peaks, dips = detect_peaks_and_dips(data, peak_type)
            logging.info(f"Peaks: {peaks[:5]}")
            logging.info(f"Dips: {dips[:5]}")

            logging.info("Generating entry signals...")
            data['buy_signal'], data['sell_signal'] = zip(*data.apply(lambda x: check_entry_conditions(x, peaks, dips, symbol), axis=1))
            logging.info("Entry signals generated.")

            buy_condition, sell_condition = generate_trade_signal(data, period, deviation_factor)

            logging.info(f"Buy Condition: {buy_condition}")
            logging.info(f"Sell Condition: {sell_condition}")

            if buy_condition or sell_condition:
                # Retrieve the latest price before executing a trade
                current_tick = get_current_data(symbol)
                logging.info(f"Latest price data for {symbol}: {current_tick}")

                trade_request = {
                    'action': 'BUY' if buy_condition else 'SELL',
                    'symbol': symbol,
                    'volume': lot_size,
                    'price': current_tick['bid'] if buy_condition else current_tick['ask'],
                    'sl': current_tick['bid'] - (1.5 * std_dev) if buy_condition else current_tick['ask'] + (1.5 * std_dev),
                    'tp': current_tick['bid'] + (2 * std_dev) if buy_condition else current_tick['ask'] - (2 * std_dev),
                    'deviation': 10,
                    'magic': 12345,
                    'comment': 'Tunnel Strategy',
                    'type': 'ORDER_TYPE_BUY' if buy_condition else 'ORDER_TYPE_SELL',
                    'type_filling': 'ORDER_FILLING_FOK',
                    'type_time': 'ORDER_TIME_GTC'
                }

                logging.info(f"Executing {'BUY' if buy_condition else 'SELL'} trade for {symbol} with trade request: {trade_request}")
                result = execute_trade(trade_request)
                if result:
                    profit = trade_request['tp'] - trade_request['price'] if buy_condition else trade_request['price'] - trade_request['tp']
                    total_profit += profit
                    current_balance += profit
                    peak_balance = max(peak_balance, current_balance)
                    drawdown = peak_balance - current_balance
                    max_drawdown = max(max_drawdown, drawdown)
                else:
                    logging.error("Trade execution failed")

            manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

        return {
            'total_profit': total_profit,
            'total_loss': total_loss,
            'max_drawdown': max_drawdown
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

#main.py contents
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from config import Config
from metatrader.connection import initialize_mt5, shutdown_mt5
from metatrader.data_retrieval import get_historical_data
from strategy.tunnel_strategy import run_strategy, calculate_ema, detect_peaks_and_dips, check_entry_conditions, check_broker_connection, check_market_open
from backtesting.backtest import run_backtest
from utils.logger import setup_logging
from utils.error_handling import handle_error
import logging
import argparse
from ui import run_ui
import os
import time

def clear_log_file():
    with open("app.log", "w"):
        pass

def check_auto_trading_enabled():
    """Check if global auto trading is enabled and log the status."""
    global_autotrading_enabled = mt5.terminal_info().trade_allowed
    if not global_autotrading_enabled:
        logging.error("Global auto trading is disabled. Please enable it manually in the MetaTrader 5 terminal.")
    else:
        logging.info("Global auto trading is enabled.")

def run_backtest_func():
    try:
        logging.info("Initializing MetaTrader5...")
        if not initialize_mt5(Config.MT5_PATH):
            raise Exception("Failed to initialize MetaTrader5")
        logging.info("MetaTrader5 initialized successfully.")

        check_auto_trading_enabled()

        for symbol in Config.SYMBOLS:
            logging.info("Running backtest...")
            start_date = datetime(2024, 6, 12)
            end_date = datetime.now()
            initial_balance = 10000
            risk_percent = Config.RISK_PER_TRADE
            stop_loss_pips = 20
            pip_value = Config.PIP_VALUE

            backtest_data = get_historical_data(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
            if backtest_data is not None and not backtest_data.empty:
                logging.info(f"Backtest data shape: {backtest_data.shape}")
                logging.info(f"Backtest data head:\n{backtest_data.head()}")
            else:
                logging.error(f"No historical data retrieved for {symbol} for backtesting")
                continue

            if len(backtest_data) < 20:
                logging.error(f"Not enough data for symbol {symbol} to perform backtest")
                continue

            backtest_data.loc[:, 'close'] = pd.to_numeric(backtest_data['close'], errors='coerce')

            try:
                run_backtest(
                    symbol=symbol,
                    data=backtest_data,
                    initial_balance=initial_balance,
                    risk_percent=risk_percent,
                    min_take_profit=Config.MIN_TP_PROFIT,
                    max_loss_per_day=Config.MAX_LOSS_PER_DAY,
                    starting_equity=Config.STARTING_EQUITY,
                    max_trades_per_day=Config.LIMIT_NO_OF_TRADES,
                    stop_loss_pips=stop_loss_pips,
                    pip_value=pip_value
                )
                logging.info("Backtest completed successfully.")
            except Exception as e:
                handle_error(e, f"An error occurred during backtesting for {symbol}")

    except Exception as e:
        error_code = mt5.last_error()
        error_message = str(e)
        handle_error(e, f"An error occurred in the run_backtest_func: {error_code} - {error_message}")

    finally:
        logging.info("Shutting down MetaTrader5...")
        shutdown_mt5()
        logging.info("MetaTrader5 connection gracefully shut down.")

def run_live_trading_func():
    try:
        logging.info("Initializing MetaTrader5...")
        if not initialize_mt5(Config.MT5_PATH):
            raise Exception("Failed to initialize MetaTrader5")
        logging.info("MetaTrader5 initialized successfully.")

        check_auto_trading_enabled()

        # Check if the account is a demo account
        account_info = mt5.account_info()
        if account_info is None:
            raise Exception("Failed to get account info")
        if account_info.server.endswith("demo"):
            logging.info("Trading on a demo account.")
        else:
            logging.info("Trading on a live account.")

        # Perform additional checks
        if not check_broker_connection():
            return

        if not check_market_open():
            return

        daily_trades = 0
        total_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        max_drawdown_reached = False
        starting_balance = Config.STARTING_EQUITY
        current_balance = starting_balance

        start_time = time.time()
        max_duration = 1 * 1800 # 10 hours

        while time.time() - start_time < max_duration:
            if max_drawdown_reached:
                logging.info("Maximum drawdown reached. Stopping trading.")
                break

            current_day = datetime.now().date()
            if daily_trades >= Config.LIMIT_NO_OF_TRADES:
                logging.info("Maximum number of trades for the day reached. Stopping trading for today.")
                time.sleep(86400)
                daily_trades = 0
                continue

            for symbol in Config.SYMBOLS:
                logging.info(f"Running live trading for {symbol}...")

                # Validate symbol availability and timeframe
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logging.error(f"Symbol {symbol} is not available.")
                    continue

                if not symbol_info.visible:
                    logging.info(f"Symbol {symbol} is not visible, attempting to make it visible.")
                    if not mt5.symbol_select(symbol, True):
                        logging.error(f"Failed to select symbol {symbol}")
                        continue

                tick_data = []
                tick_start_time = time.time()

                while len(tick_data) < 200:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        logging.warning(f"Failed to retrieve tick data for {symbol}.")
                        time.sleep(1)
                        continue

                    tick_data.append({
                        'time': datetime.fromtimestamp(tick.time),
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'last': tick.last
                    })

                    time.sleep(1)

                tick_end_time = time.time()
                elapsed_time = tick_end_time - tick_start_time
                logging.info(f"Collected 200 ticks in {elapsed_time:.2f} seconds.")

                df = pd.DataFrame(tick_data)
                logging.info(f"Dataframe created with tick data: {df.tail()}")

                # Ensure DataFrame has all necessary columns
                if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
                    df['high'] = df['bid']
                    df['low'] = df['ask']
                    df['close'] = df['last']

                std_dev = df['close'].rolling(window=20).std().iloc[-1]  # Added this line

                try:
                    result = run_strategy(
                        symbols=[symbol],
                        mt5_init=mt5,
                        timeframe=mt5.TIMEFRAME_M1,
                        lot_size=0.01,
                        min_take_profit=Config.MIN_TP_PROFIT,
                        max_loss_per_day=Config.MAX_LOSS_PER_DAY,
                        starting_equity=current_balance,
                        max_trades_per_day=Config.LIMIT_NO_OF_TRADES,
                        run_backtest=False,
                        data=df,
                        std_dev=std_dev  # Pass std_dev to run_strategy
                    )

                    if result is None:
                        raise ValueError("run_strategy returned None. Check the function implementation.")

                    total_profit += result.get('total_profit', 0.0)
                    total_loss += result.get('total_loss', 0.0)
                    current_balance += result.get('total_profit', 0.0)

                    max_drawdown = result.get('max_drawdown', 0.0)
                    if max_drawdown >= Config.MAX_DRAWDOWN:
                        max_drawdown_reached = True
                        logging.info(f"Maximum drawdown of {Config.MAX_DRAWDOWN} reached. Stopping trading.")
                        break

                    daily_trades += 1
                    total_trades += 1
                    logging.info(f"Live trading iteration completed for {symbol}. Total trades today: {daily_trades}")
                    logging.info(f"Current Balance: {current_balance:.2f}")

                except Exception as e:
                    logging.error(f"An error occurred while running strategy for {symbol}: {e}")

                time.sleep(60)

            if time.time() - start_time >= max_duration:
                logging.info("Maximum duration reached. Stopping trading.")
                break

    except Exception as e:
        error_code = mt5.last_error()
        error_message = str(e)
        handle_error(e, f"An error occurred in the run_live_trading_func: {error_code} - {error_message}")

    finally:
        logging.info("Shutting down MetaTrader5...")
        shutdown_mt5()
        logging.info("MetaTrader5 connection gracefully shut down.")

        logging.info("Summary of Trading Session:")
        logging.info(f"Total trades: {total_trades}")
        logging.info(f"Starting balance: {starting_balance:.2f}")
        logging.info(f"Ending balance: {current_balance:.2f}")
        logging.info(f"Total profit: {total_profit:.2f}")
        logging.info(f"Total loss: {total_loss:.2f}")

def open_log_file():
    import subprocess
    log_file_path = os.path.abspath("app.log")
    if os.name == "nt":
        os.startfile(log_file_path)
    elif os.name == "posix":
        subprocess.call(["open", log_file_path])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", action="store_true", help="Run the UI")
    args = parser.parse_args()

    try:
        setup_logging()
        logging.info("STARTING APPLICATION")

        logging.info("LOGGING ALL THE CONFIG SETTINGS")
        Config.log_config()

        if args.ui:
            run_ui(run_backtest_func, run_live_trading_func, clear_log_file, open_log_file)
        else:
            print("Choose an option:")
            print("1. Run Backtesting")
            print("2. Run Live Trading")
            choice = input("Enter your choice (1 or 2): ")

            if choice == "1":
                run_backtest_func()
            elif choice == "2":
                run_live_trading_func()
            else:
                print("Invalid choice. Exiting...")

    except Exception as e:
        error_code = mt5.last_error()
        error_message = str(e)
        handle_error(e, f"An error occurred in the main function: {error_code} - {error_message}")

if __name__ == '__main__':
    main()


#config contents:
import os
from dotenv import load_dotenv
from utils.error_handling import handle_error, critical_error
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
dotenv_path = os.path.join(project_dir, '.env')

def reload_env():
    load_dotenv(dotenv_path, override=True)

reload_env()

class Config:
    MT5_LOGIN = os.getenv("MT5_LOGIN")
    if not MT5_LOGIN:
        raise ValueError("MT5_LOGIN environment variable is not set.")

    MT5_PASSWORD = os.getenv("MT5_PASSWORD")
    if not MT5_PASSWORD:
        raise ValueError("MT5_PASSWORD environment variable is not set.")

    MT5_SERVER = os.getenv("MT5_SERVER")
    if not MT5_SERVER:
        raise ValueError("MT5_SERVER environment variable is not set.")

    MT5_PATH = os.getenv("MT5_PATH")
    if not MT5_PATH:
        raise ValueError("MT5_PATH environment variable is not set.")

    MT5_TIMEFRAME = os.getenv("MT5_TIMEFRAME")
    if MT5_TIMEFRAME not in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
        raise ValueError(f"Invalid MT5_TIMEFRAME value: {MT5_TIMEFRAME}. Expected values: M1, M5, M15, M30, H1, H4, D1.")

    SYMBOLS = os.getenv("SYMBOLS")
    if SYMBOLS:
        SYMBOLS = SYMBOLS.split(",")
    else:
        raise ValueError("SYMBOLS environment variable is not set.")

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_IDS = os.getenv("TELEGRAM_IDS")
    if TELEGRAM_TOKEN and TELEGRAM_IDS:
        TELEGRAM_IDS = TELEGRAM_IDS.split(",")
    else:
        TELEGRAM_TOKEN = None
        TELEGRAM_IDS = None

    try:
        MIN_TP_PROFIT = float(os.getenv("MIN_TP_PROFIT", 50.0))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid MIN_TP_PROFIT value: {os.getenv('MIN_TP_PROFIT')}. Expected a numeric value.")

    try:
        MAX_LOSS_PER_DAY = float(os.getenv("MAX_LOSS_PER_DAY", 1000.0))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid MAX_LOSS_PER_DAY value: {os.getenv('MAX_LOSS_PER_DAY')}. Expected a numeric value.")

    try:
        STARTING_EQUITY = float(os.getenv("STARTING_EQUITY", 10000.0))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid STARTING_EQUITY value: {os.getenv('STARTING_EQUITY')}. Expected a numeric value.")

    try:
        LIMIT_NO_OF_TRADES = int(os.getenv("LIMIT_NO_OF_TRADES", 5))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid LIMIT_NO_OF_TRADES value: {os.getenv('LIMIT_NO_OF_TRADES')}. Expected an integer value.")

    try:
        RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
    except ValueError:
        raise ValueError(f"Invalid RISK_PER_TRADE value: {os.getenv('RISK_PER_TRADE')}. Expected a numeric value.")

    if not 0 < RISK_PER_TRADE <= 1:
        raise ValueError(f"RISK_PER_TRADE value must be between 0 and 1. Current value: {RISK_PER_TRADE}")

    try:
        PIP_VALUE = float(os.getenv("PIP_VALUE", 1))
    except ValueError:
        raise ValueError(f"Invalid PIP_VALUE value: {os.getenv('PIP_VALUE')}. Expected a numeric value.")

    try:
        MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", 0.2))
    except ValueError:
        raise ValueError(f"Invalid MAX_DRAWDOWN value: {os.getenv('MAX_DRAWDOWN')}. Expected a numeric value.")

    @classmethod
    def validate(cls):
        try:
            required_vars = [
                'MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER', 'MT5_PATH',
                'MT5_TIMEFRAME', 'SYMBOLS'
            ]
            for var in required_vars:
                if not getattr(cls, var, None):
                    raise ValueError(f"Missing required environment variable: {var}")

            numeric_vars = ['MIN_TP_PROFIT', 'MAX_LOSS_PER_DAY', 'STARTING_EQUITY', 'RISK_PER_TRADE', 'PIP_VALUE', 'MAX_DRAWDOWN']
            for var in numeric_vars:
                if not isinstance(getattr(cls, var, None), (int, float)):
                    raise ValueError(f"Invalid value for {var}. Expected a numeric value.")

            if not isinstance(cls.LIMIT_NO_OF_TRADES, int):
                raise ValueError(f"Invalid value for LIMIT_NO_OF_TRADES. Expected an integer value.")

        except ValueError as e:
            handle_error(e, "Configuration validation failed")
            critical_error(e, "Invalid configuration settings")

    @classmethod
    def log_config(cls):
        for attr, value in cls.__dict__.items():
            if not callable(value) and not attr.startswith("__") and not isinstance(value, classmethod):
                logging.info(f"{attr}: {value}")

try:
    Config.validate()
except Exception as e:
    handle_error(e, "Error occurred during configuration validation")
    raise


#.env contents:

# MetaTrader 5 Configuration
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_TIMEFRAME=H1

# Trading Symbols
SYMBOLS=EURUSD

# Telegram Bot Configuration (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_IDS=telegram_user_id_1,telegram_user_id_2

# Trading Parameters
MIN_TP_PROFIT=50.0
MAX_LOSS_PER_DAY=1000.0
STARTING_EQUITY=10000.0
LIMIT_NO_OF_TRADES=12
RISK_PER_TRADE=0.02
PIP_VALUE=0.0001
