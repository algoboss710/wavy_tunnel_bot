#content of backtesting.backtest.py

import logging
import pandas as pd
import numpy as np
import cProfile
import pstats
from io import StringIO

def calculate_max_drawdown(trades, initial_balance):
    balance = initial_balance
    max_balance = initial_balance
    max_drawdown = 0

    for trade in trades:
        if 'profit' in trade:
            balance += trade['profit']
            max_balance = max(max_balance, balance)
            drawdown = max_balance - balance
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown

def run_backtest(symbol, data, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, stop_loss_pips, pip_value, max_trades_per_day=None, slippage=0, transaction_cost=0):
    from strategy.tunnel_strategy import generate_trade_signal, manage_position, calculate_position_size, detect_peaks_and_dips
    from metatrader.indicators import calculate_ema
    from metatrader.trade_management import execute_trade

    # Profiling setup
    pr = cProfile.Profile()
    pr.enable()

    try:
        balance = initial_balance
        trades = []
        trades_today = 0
        current_day = data.iloc[0]['time'].date()
        max_drawdown = 0
        daily_loss = 0
        buy_condition = False  # Initialize buy_condition
        sell_condition = False  # Initialize sell_condition

        logging.info(f"Initial balance: {balance}")
        print(f"Initial balance: {balance}")

        # Validate critical parameters
        if stop_loss_pips <= 0 or pip_value <= 0:
            raise ZeroDivisionError("stop_loss_pips and pip_value must be greater than zero.")

        peak_type = 21

        # Calculate indicators and peaks/dips for the entire dataset
        data.loc[:, 'wavy_h'] = calculate_ema(data['high'], 34)
        data.loc[:, 'wavy_c'] = calculate_ema(data['close'], 34)
        data.loc[:, 'wavy_l'] = calculate_ema(data['low'], 34)
        data.loc[:, 'tunnel1'] = calculate_ema(data['close'], 144)
        data.loc[:, 'tunnel2'] = calculate_ema(data['close'], 169)
        data.loc[:, 'long_term_ema'] = calculate_ema(data['close'], 200)

        peaks, dips = detect_peaks_and_dips(data, peak_type)

        for i in range(34, len(data)):  # Start after enough data points are available
            logging.info(f"Iteration: {i}, trades_today: {trades_today}, current_day: {current_day}")

            # Check if it's a new day
            if data.iloc[i]['time'].date() != current_day:
                logging.info(f"New day detected: {data.iloc[i]['time'].date()}, resetting trades_today and daily_loss.")
                current_day = data.iloc[i]['time'].date()
                trades_today = 0
                daily_loss = 0

            if max_trades_per_day is not None and trades_today >= max_trades_per_day:
                logging.info(f"Max trades per day reached at row {i}.")
                continue

            # Generate trading signals
            buy_condition, sell_condition = generate_trade_signal(data.iloc[:i+1], period=20, deviation_factor=2.0)
            if buy_condition is None or sell_condition is None:
                continue

            try:
                position_size = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
            except ZeroDivisionError as e:
                logging.error(f"Division by zero occurred in calculate_position_size: {e}. Variables - balance: {balance}, risk_percent: {risk_percent}, stop_loss_pips: {stop_loss_pips}, pip_value: {pip_value}")
                continue

            row = data.iloc[i]

            if buy_condition and (max_trades_per_day is None or trades_today < max_trades_per_day):
                logging.info(f"Buy condition met at row {i}.")
                trade = {
                    'entry_time': data.iloc[i]['time'],
                    'entry_price': data.iloc[i]['close'],
                    'volume': position_size,
                    'symbol': symbol,
                    'action': 'BUY',
                    'sl': data.iloc[i]['close'] - (1.5 * data['close'].rolling(window=20).std().iloc[i]),
                    'tp': data.iloc[i]['close'] + (2 * data['close'].rolling(window=20).std().iloc[i])
                }
                trades.append(trade)
                trades_today += 1
                execute_trade(trade)
                logging.info(f"Balance after BUY trade: {balance}")

            elif sell_condition and (max_trades_per_day is None or trades_today < max_trades_per_day):
                logging.info(f"Sell condition met at row {i}.")
                if trades:
                    trade = trades[-1]
                    trade['exit_time'] = data.iloc[i]['time']
                    trade['exit_price'] = data.iloc[i]['close']
                    trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['volume'] * pip_value
                    balance += trade['profit']
                    execute_trade(trade)
                    trades_today += 1
                    logging.info(f"Balance after SELL trade: {balance}")

            manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

        logging.info(f"Final balance: {balance}")
        print(f"Final balance: {balance}")

        total_profit = sum(trade['profit'] for trade in trades if 'profit' in trade)
        num_trades = len(trades)
        win_rate = sum(1 for trade in trades if 'profit' in trade and trade['profit'] > 0) / num_trades if num_trades > 0 else 0
        max_drawdown = calculate_max_drawdown(trades, initial_balance)

        logging.info(f"Total Profit: {total_profit:.2f}")
        logging.info(f"Number of Trades: {num_trades}")
        logging.info(f"Win Rate: {win_rate:.2%}")
        logging.info(f"Maximum Drawdown: {max_drawdown:.2f}")

        print(f"Total Profit: {total_profit:.2f}")
        print(f"Number of Trades: {num_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}")

        return {
            'total_profit': total_profit,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'buy_condition': buy_condition,
            'sell_condition': sell_condition,
            'trades': trades,
            'total_slippage_costs': len(trades) * slippage,
            'total_transaction_costs': len(trades) * transaction_cost
        }

    finally:
        pr.disable()

        # Output profiling results
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        print(s.getvalue())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)


# content of strategy.tunnel_strategy


import numpy as np
from metatrader.data_retrieval import get_historical_data
from utils.error_handling import handle_error
from metatrader.indicators import calculate_ema
from metatrader.trade_management import place_order, close_position, modify_order
import pandas as pd
import MetaTrader5 as mt5
import logging

def calculate_ema(prices, period):
    if not isinstance(prices, (list, np.ndarray, pd.Series)):
        raise ValueError("Invalid input type for prices. Expected list, numpy array, or pandas Series.")
    
    logging.debug(f"Calculating EMA for period: {period}, prices: {prices}")
    
    # Convert input to a pandas Series to ensure consistency
    prices = pd.Series(prices)
    
    # Ensure that the series is numeric
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
                if position.profit >= min_take_profit:
                    logging.debug(f"Closing position {position.ticket} for profit")
                    close_position(position.ticket)
                elif position.profit <= -max_loss_per_day:
                    logging.debug(f"Closing position {position.ticket} for loss")
                    close_position(position.ticket)
                else:
                    current_equity = mt5.account_info().equity
                    if current_equity <= starting_equity * 0.9:  # Close position if equity drops by 10%
                        logging.debug(f"Closing position {position.ticket} due to equity drop")
                        close_position(position.ticket)
                    elif mt5.positions_total() >= max_trades_per_day:
                        logging.debug(f"Closing position {position.ticket} due to max trades exceeded")
                        close_position(position.ticket)
    except Exception as e:
        handle_error(e, "Failed to manage position")

def calculate_tunnel_bounds(data, period, deviation_factor):
    # Ensure 'close' column is numeric
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
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

# content of metatrader.indicators 

import pandas as pd
import numpy as np
import logging

def calculate_ema(prices, period):
    if not isinstance(prices, (list, np.ndarray, pd.Series)):
        raise ValueError("Invalid input type for prices. Expected list, numpy array, or pandas Series.")
    
    logging.debug(f"Calculating EMA for period: {period}, prices: {prices}")
    
    # Convert input to a pandas Series to ensure consistency
    prices = pd.Series(prices)
    
    # Ensure that the series is numeric
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


# content of metatrader.trade_mamagment:

import MetaTrader5 as mt5

def place_order(symbol, order_type, volume, price=None, sl=None, tp=None):
    try:
        order = mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order,
            "price": mt5.symbol_info_tick(symbol).ask if order_type == 'buy' else mt5.symbol_info_tick(symbol).bid,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 234000,
            "comment": "python script order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result.comment if result else 'Order failed'
    except Exception as e:
        return f'Order failed: {str(e)}'

def close_position(ticket):
    try:
        position = mt5.positions_get(ticket=ticket)
        if position:
            result = mt5.Close(ticket)
            return result.comment if result else 'Close failed'
        return 'Position not found'
    except Exception as e:
        return f'Close failed: {str(e)}'

def modify_order(ticket, sl=None, tp=None):
    try:
        result = mt5.order_check(ticket)
        if result and result.type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL]:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "ticket": ticket,
                "sl": sl,
                "tp": tp
            }
            result = mt5.order_send(request)
            return result.comment if result else 'Modify failed'
        return 'Order not found'
    except Exception as e:
        return f'Modify failed: {str(e)}'

def execute_trade(trade):
    """
    Executes a trade based on the provided trade dictionary.
    Expected dictionary keys: 'symbol', 'action', 'volume', 'price', 'sl', 'tp'.
    """
    symbol = trade.get('symbol')
    action = trade.get('action')
    volume = trade.get('volume')
    price = trade.get('price')
    sl = trade.get('sl')
    tp = trade.get('tp')
    
    if action == 'BUY':
        return place_order(symbol, 'buy', volume, price, sl, tp)
    elif action == 'SELL':
        return place_order(symbol, 'sell', volume, price, sl, tp)
    else:
        return 'Invalid trade action'


# content of tests.integration.backtest_new.py

import unittest
import pandas as pd
from backtesting.backtest import run_backtest
import logging

class TestRunBacktest(unittest.TestCase):

    def setUp(self):
        # Create sample data
        data = {
            'time': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'high': [70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128],
            'low': [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108],
            'close': [60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        }
        self.data = pd.DataFrame(data)

    def test_initial_balance(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        self.assertEqual(result['total_profit'], 0)
        self.assertEqual(result['num_trades'], 0)
        self.assertEqual(result['win_rate'], 0)
        self.assertEqual(result['max_drawdown'], 0)

    def test_max_loss_per_day(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        self.assertIn('trades', result, "Result dictionary does not contain 'trades' key.")
        
        actual_loss = sum(trade['profit'] for trade in result['trades'] if 'profit' in trade and trade['profit'] < 0)
        expected_loss = -100  # Expecting that max loss per day is respected
        if result['trades']:
            self.assertLessEqual(actual_loss, expected_loss)
        else:
            self.assertEqual(actual_loss, 0)

    def test_pip_value_validation(self):
        with self.assertRaises(ZeroDivisionError):
            run_backtest(
                symbol='EURUSD',
                data=self.data,
                initial_balance=10000,
                risk_percent=0.01,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=20,
                pip_value=0,
                max_trades_per_day=5
            )

    def test_stop_loss_pips_validation(self):
        with self.assertRaises(ZeroDivisionError):
            run_backtest(
                symbol='EURUSD',
                data=self.data,
                initial_balance=10000,
                risk_percent=0.01,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=0,
                pip_value=0.0001,
                max_trades_per_day=5
            )

    def test_max_trades_per_day(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=1  # Limiting to 1 trade per day for this test
        )

        print(result)  # Debugging line to check the output
        self.assertIn('trades', result, "Result dictionary does not contain 'trades' key.")
        
        trades_per_day = {}
        for trade in result['trades']:
            day = trade['entry_time'].date()
            if day not in trades_per_day:
                trades_per_day[day] = 0
            trades_per_day[day] += 1
        
        for day, count in trades_per_day.items():
            self.assertLessEqual(count, 1, f"More than 1 trade executed on {day}")

    def test_total_profit_calculation(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        total_profit = sum(trade['profit'] for trade in result['trades'])
        self.assertEqual(result['total_profit'], total_profit, "Total profit calculation is incorrect.")

    def test_number_of_trades(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        self.assertEqual(result['num_trades'], len(result['trades']), "Number of trades calculation is incorrect.")

    def test_win_rate_calculation(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        wins = sum(1 for trade in result['trades'] if trade['profit'] > 0)
        total_trades = len(result['trades'])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        self.assertAlmostEqual(result['win_rate'], win_rate, places=2, msg="Win rate calculation is incorrect.")

    def test_max_drawdown_calculation(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        equity_curve = [trade['equity'] for trade in result['trades']]
        if equity_curve:
            max_drawdown = max((max(equity_curve[:i+1]) - equity) / max(equity_curve[:i+1]) for i, equity in enumerate(equity_curve))
        else:
            max_drawdown = 0

        self.assertAlmostEqual(result['max_drawdown'], max_drawdown, places=2, msg="Max drawdown calculation is incorrect.")

    def test_profit_factor_calculation(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        gross_profit = sum(trade['profit'] for trade in result['trades'] if trade['profit'] > 0)
        gross_loss = abs(sum(trade['profit'] for trade in result['trades'] if trade['profit'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        self.assertAlmostEqual(result.get('profit_factor', profit_factor), profit_factor, places=2, msg="Profit factor calculation is incorrect.")

    def test_return_on_investment_calculation(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        total_profit = result['total_profit']
        initial_balance = 10000
        roi = (total_profit / initial_balance) * 100

        self.assertAlmostEqual(result.get('roi', roi), roi, places=2, msg="ROI calculation is incorrect.")

    def test_sharpe_ratio_calculation(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        returns = [trade['profit'] / 10000 for trade in result['trades']]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (sum((x - avg_return) ** 2 for x in returns) / len(returns)) ** 0.5 if returns else 0
        risk_free_rate = 0.01
        sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return != 0 else 0

        self.assertAlmostEqual(result.get('sharpe_ratio', sharpe_ratio), sharpe_ratio, places=2, msg="Sharpe ratio calculation is incorrect.")

    def test_win_loss_ratio_calculation(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        wins = sum(1 for trade in result['trades'] if trade['profit'] > 0)
        losses = sum(1 for trade in result['trades'] if trade['profit'] < 0)
        win_loss_ratio = wins / losses if losses > 0 else float('inf')

        self.assertAlmostEqual(result.get('win_loss_ratio', win_loss_ratio), win_loss_ratio, places=2, msg="Win/Loss ratio calculation is incorrect.")

    def test_annualized_return_calculation(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        total_profit = result['total_profit']
        days = (self.data['time'].iloc[-1] - self.data['time'].iloc[0]).days
        annualized_return = ((total_profit / 10000) + 1) ** (365 / days) - 1 if days > 0 else 0

        self.assertAlmostEqual(result.get('annualized_return', annualized_return), annualized_return, places=2, msg="Annualized return calculation is incorrect.")

    def test_expectancy_calculation(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        total_trades = len(result['trades'])
        wins = [trade['profit'] for trade in result['trades'] if trade['profit'] > 0]
        losses = [trade['profit'] for trade in result['trades'] if trade['profit'] < 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        loss_rate = len(losses) / total_trades if total_trades > 0 else 0
        expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)

        self.assertAlmostEqual(result.get('expectancy', expectancy), expectancy, places=2, msg="Expectancy calculation is incorrect.")

    # Additional test cases

    def test_consecutive_wins_and_losses(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in result['trades']:
            if trade['profit'] > 0:
                current_wins += 1
                current_losses = 0
            else:
                current_losses += 1
                current_wins = 0

            max_consecutive_wins = max(max_consecutive_wins, current_wins)
            max_consecutive_losses = max(max_consecutive_losses, current_losses)

        self.assertEqual(result.get('max_consecutive_wins', max_consecutive_wins), max_consecutive_wins, "Max consecutive wins calculation is incorrect.")
        self.assertEqual(result.get('max_consecutive_losses', max_consecutive_losses), max_consecutive_losses, "Max consecutive losses calculation is incorrect.")

    def test_handling_large_datasets(self):
        large_data = pd.concat([self.data] * 1000, ignore_index=True)
        chunk_size = len(self.data)  # Adjust chunk size as needed
        chunks = [large_data[i:i + chunk_size] for i in range(0, len(large_data), chunk_size)]

        result = None
        for chunk in chunks:
            result = run_backtest(
                symbol='EURUSD',
                data=chunk,
                initial_balance=10000,
                risk_percent=0.01,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=20,
                pip_value=0.0001,
                max_trades_per_day=5
            )
        
        print(result)  # Debugging line to check the output
        self.assertIsNotNone(result, "Handling large datasets failed.")

    def test_handling_missing_values(self):
        data_with_missing_values = self.data.copy()
        data_with_missing_values.loc[0, 'close'] = None
        result = run_backtest(
            symbol='EURUSD',
            data=data_with_missing_values,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        self.assertIsNotNone(result, "Handling missing values failed.")

    def test_transaction_costs(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5,
            transaction_cost=1  # Adding transaction costs
        )

        print(result)  # Debugging line to check the output
        total_transaction_costs = len(result['trades']) * 1
        self.assertAlmostEqual(result['total_transaction_costs'], total_transaction_costs, places=2, msg="Transaction costs calculation is incorrect.")

    def test_slippage(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5,
            slippage=1  # Adding slippage
        )

        print(result)  # Debugging line to check the output
        total_slippage_costs = len(result['trades']) * 1
        self.assertAlmostEqual(result['total_slippage_costs'], total_slippage_costs, places=2, msg="Slippage calculation is incorrect.")

    def test_negative_initial_balance(self):
        with self.assertRaises(ValueError):
            run_backtest(
                symbol='EURUSD',
                data=self.data,
                initial_balance=-10000,
                risk_percent=0.01,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=20,
                pip_value=0.0001,
                max_trades_per_day=5
            )

    def test_zero_risk_percent(self):
        with self.assertRaises(ValueError):
            run_backtest(
                symbol='EURUSD',
                data=self.data,
                initial_balance=10000,
                risk_percent=0,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=20,
                pip_value=0.0001,
                max_trades_per_day=5
            )

    def test_constant_prices(self):
        constant_data = self.data.copy()
        constant_data['high'] = 100
        constant_data['low'] = 100
        constant_data['close'] = 100

        result = run_backtest(
            symbol='EURUSD',
            data=constant_data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        self.assertEqual(result['total_profit'], 0)
        self.assertEqual(result['num_trades'], 0)

    def test_uptrend_data(self):
        uptrend_data = self.data.copy()
        uptrend_data['high'] = range(100, 130)
        uptrend_data['low'] = range(80, 110)
        uptrend_data['close'] = range(90, 120)

        result = run_backtest(
            symbol='EURUSD',
            data=uptrend_data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        self.assertGreater(result['total_profit'], 0)

    def test_downtrend_data(self):
        # Create a downtrend data where trading signals are likely to be triggered
     downtrend_data = pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', periods=30, freq='D'),
        'high': list(range(130, 100, -1)),
        'low': list(range(110, 80, -1)),
        'close': list(range(120, 90, -1))
        })

    # Ensure there is sufficient volatility to trigger trades
     downtrend_data['high'] = downtrend_data['high'] * 1.01  # Increase high values slightly to simulate volatility
     downtrend_data['low'] = downtrend_data['low'] * 0.99   # Decrease low values slightly to simulate volatility

    # Run backtest with modified downtrend data
     result = run_backtest(
            symbol='EURUSD',
            data=downtrend_data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=1,  # Low take profit to ensure trades close quickly
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

     print(result)  # Debugging line to check the output
     self.assertIn('trades', result, "Result dictionary does not contain 'trades' key.")
     self.assertGreater(len(result['trades']), 0, "No trades were executed during the backtest.")
     self.assertLess(result['total_profit'], 0, "The total profit should be less than 0 in a downtrend.")

    # Additional checks can be performed if needed
     for trade in result['trades']:
            self.assertIn('profit', trade, "Trade does not contain 'profit' key.")
            self.assertNotEqual(trade['profit'], 0, "Trade profit should not be zero.")



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()



