import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict
import json
import os
import concurrent.futures

# Create test_logs folder if it doesn't exist
os.makedirs('test_logs_backtest', exist_ok=True)

# Set up logging
def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

success_logger = setup_logger('successful_trades', 'test_logs_backtest/successful_trades.log')
failure_logger = setup_logger('failed_trades', 'test_logs_backtest/failed_trades.log', logging.ERROR)
summary_logger = setup_logger('summary', 'test_logs_backtest/summary.log')

# Load configuration from JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

GLOBAL_START_DATE = datetime.strptime(config['global_settings']['start_date'], '%Y-%m-%d')
GLOBAL_END_DATE = datetime.strptime(config['global_settings']['end_date'], '%Y-%m-%d')

# Connect to MetaTrader 5
if not mt5.initialize():
    logging.error("MetaTrader5 initialization failed")
    mt5.shutdown()
    quit()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period):
    tr = pd.concat([high - low,
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def get_historical_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        logging.warning(f"Failed to retrieve data for {symbol} on {timeframe} timeframe")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    logging.info(f"Retrieved {len(df)} bars for {symbol} on {timeframe} timeframe")
    return df

def calculate_min_gap_second(symbol):
    currency_gaps = {
        "USD": 15,
        "EUR": 15,
        "JPY": 650,
        "GBP": 50,
        "CHF": 15,
        "AUD": 15
    }
    base_currency = symbol[:3]
    return currency_gaps.get(base_currency, 15)  # Default to 15 if currency not found

def calculate_peaks_dips(df, lookback, threshold):
    highs = df['high'].rolling(window=lookback, center=True).max()
    lows = df['low'].rolling(window=lookback, center=True).min()
    df['peak'] = df['high'].where((df['high'] > highs.shift(1)) & (df['high'] > highs.shift(-1)) & (df['high'] - df['low'].rolling(lookback).min() > threshold), np.nan)
    df['dip'] = df['low'].where((df['low'] < lows.shift(1)) & (df['low'] < lows.shift(-1)) & (df['high'].rolling(lookback).max() - df['low'] > threshold), np.nan)
    return df

def wavy_tunnel_strategy(df, params):
    df['wavy_h'] = calculate_ema(df['high'], params['wavy_ema'])
    df['wavy_c'] = calculate_ema(df['close'], params['wavy_ema'])
    df['wavy_l'] = calculate_ema(df['low'], params['wavy_ema'])
    df['tunnel1'] = calculate_ema(df['close'], params['tunnel_ema1'])
    df['tunnel2'] = calculate_ema(df['close'], params['tunnel_ema2'])

    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], params['atr_period'])
    df['threshold'] = df['atr'] * params['atr_multiplier']

    max_wavy = df[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)
    min_wavy = df[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)
    max_tunnel = df[['tunnel1', 'tunnel2']].max(axis=1)
    min_tunnel = df[['tunnel1', 'tunnel2']].min(axis=1)

    # Primary strategy conditions
    df['long_condition'] = (df['open'] > max_wavy + df['threshold']) & (min_wavy > max_tunnel)
    df['short_condition'] = (df['open'] < min_wavy - df['threshold']) & (max_wavy < min_tunnel)

    # Secondary strategy conditions
    if params.get('enable_second_strategy', False):
        min_gap_second = calculate_min_gap_second(params['currency_pair']) * mt5.symbol_info(params['currency_pair']).point
        df['second_long_condition'] = (df['close'] > max_wavy) & (df['close'] < min_tunnel) & \
                                      (min_tunnel - df['close'] >= min_gap_second)
        df['second_short_condition'] = (df['close'] < min_wavy) & (df['close'] > max_tunnel) & \
                                       (df['close'] - max_tunnel >= min_gap_second)

        # Apply MaxAllowIntoZone
        df['long_zone_size'] = min_tunnel - max_wavy
        df['short_zone_size'] = min_wavy - max_tunnel
        df['second_long_condition'] &= (df['close'] - max_wavy) / df['long_zone_size'] <= params.get('max_allow_into_zone', 0.25)
        df['second_short_condition'] &= (min_wavy - df['close']) / df['short_zone_size'] <= params.get('max_allow_into_zone', 0.25)
    else:
        df['second_long_condition'] = False
        df['second_short_condition'] = False

    if params.get('apply_rsi_filter', False):
        df['rsi'] = calculate_rsi(df, params.get('rsi_period', 14))
        df['long_condition'] &= df['rsi'] < params.get('rsi_upper', 70)
        df['short_condition'] &= df['rsi'] > params.get('rsi_lower', 30)
        df['second_long_condition'] &= df['rsi'] < params.get('rsi_upper', 70)
        df['second_short_condition'] &= df['rsi'] > params.get('rsi_lower', 30)

    df['exit_long'] = df['close'] < min_wavy
    df['exit_short'] = df['close'] > max_wavy

    # Calculate peaks and dips
    peak_dip_lookback = params.get('peak_dip_lookback', 100)  # Default to 100 if not provided
    peak_dip_threshold = params.get('peak_dip_threshold', 0.001)  # Default to 0.001 if not provided
    df = calculate_peaks_dips(df, peak_dip_lookback, peak_dip_threshold)

    return df

def simulate_trade(entry_price, exit_price, order_type, lot_size):
    if order_type == 'buy':
        profit = (exit_price - entry_price) * lot_size * 100000  # Adjust for lot size and pip value
    elif order_type == 'sell':
        profit = (entry_price - exit_price) * lot_size * 100000
    return profit

def backtest(symbol_config):
    symbol = symbol_config['currency_pair']
    timeframe = getattr(mt5, f"TIMEFRAME_{symbol_config['timeframe']}")
    logger = setup_logger(f'{symbol}_{timeframe}', f'test_logs_backtest/{symbol}_{timeframe}.log')

    df = get_historical_data(symbol, timeframe, GLOBAL_START_DATE, GLOBAL_END_DATE)
    if df is None:
        logger.warning(f"No data for {symbol} on {timeframe} timeframe, skipping backtest")
        return None

    df = wavy_tunnel_strategy(df, symbol_config)
    logger.info(f"Starting backtest for {symbol} on {timeframe} timeframe")

    balance = 10000  # Starting balance for each symbol
    position = None
    open_tp_orders = []
    trades_analyzed = 0
    trades_executed_long = 0
    trades_executed_short = 0
    trades_executed_second_long = 0
    trades_executed_second_short = 0
    profit_long = 0
    profit_short = 0
    profit_second_long = 0
    profit_second_short = 0
    for i in range(1, len(df)):
        row = df.iloc[i]
        trades_analyzed += 1

        # Check and execute take profit orders
        if open_tp_orders:
            executed_tps = []
            for tp in open_tp_orders:
                if (position['type'] == 'buy' and row['high'] >= tp['price']) or \
                   (position['type'] == 'sell' and row['low'] <= tp['price']):
                    profit = simulate_trade(position['entry_price'], tp['price'], position['type'], tp['quantity'])
                    logger.info(f"Take profit executed at {tp['price']} | Profit: {profit}")
                    if position['strategy'] == 'primary':
                        if position['type'] == 'buy':
                            profit_long += profit
                        else:
                            profit_short += profit
                    else:
                        if position['type'] == 'buy':
                            profit_second_long += profit
                        else:
                            profit_second_short += profit
                    balance += profit
                    executed_tps.append(tp)

            for tp in executed_tps:
                open_tp_orders.remove(tp)

            if not open_tp_orders:
                position = None

        if position is None:
            # Entry conditions
            if row['long_condition'] or row['second_long_condition']:
                position = {'type': 'buy', 'entry_price': row['open'], 'entry_time': row['time'],
                            'strategy': 'primary' if row['long_condition'] else 'secondary'}

                # Set take profit levels
                if position['strategy'] == 'primary':
                    last_dip = df['dip'].iloc[max(0, i-symbol_config['peak_dip_lookback']):i].last_valid_index()
                    if last_dip is not None:
                        tp_distance = row['open'] - df['dip'].loc[last_dip]
                    else:
                        tp_distance = row['atr'] * symbol_config['tp_atr_multiplier']
                else:
                    tp_distance = df['tunnel1'].iloc[i] - row['open']

                for level, quantity in zip(symbol_config['tp_levels'], symbol_config['tp_quantities']):
                    tp_price = row['open'] + (tp_distance * level)
                    open_tp_orders.append({'price': tp_price, 'quantity': symbol_config['lot_size'] * quantity})

                logger.info(f"Opening {position['strategy']} long position at {row['open']} with {len(open_tp_orders)} TP orders")
                if position['strategy'] == 'primary':
                    trades_executed_long += 1
                else:
                    trades_executed_second_long += 1

            elif row['short_condition'] or row['second_short_condition']:
                position = {'type': 'sell', 'entry_price': row['open'], 'entry_time': row['time'],
                            'strategy': 'primary' if row['short_condition'] else 'secondary'}

                # Set take profit levels
                if position['strategy'] == 'primary':
                    last_peak = df['peak'].iloc[max(0, i-symbol_config['peak_dip_lookback']):i].last_valid_index()
                    if last_peak is not None:
                        tp_distance = df['peak'].loc[last_peak] - row['open']
                    else:
                        tp_distance = row['atr'] * symbol_config['tp_atr_multiplier']
                else:
                    tp_distance = row['open'] - df['tunnel2'].iloc[i]

                for level, quantity in zip(symbol_config['tp_levels'], symbol_config['tp_quantities']):
                    tp_price = row['open'] - (tp_distance * level)
                    open_tp_orders.append({'price': tp_price, 'quantity': symbol_config['lot_size'] * quantity})

                logger.info(f"Opening {position['strategy']} short position at {row['open']} with {len(open_tp_orders)} TP orders")
                if position['strategy'] == 'primary':
                    trades_executed_short += 1
                else:
                    trades_executed_second_short += 1

        # Exit conditions
        if position:
            if (position['type'] == 'buy' and row['exit_long']) or \
               (position['type'] == 'sell' and row['exit_short']):
                profit = simulate_trade(position['entry_price'], row['close'], position['type'], symbol_config['lot_size'])
                logger.info(f"Closing {position['strategy']} {position['type']} position at {row['close']} | Profit: {profit}")
                if position['strategy'] == 'primary':
                    if position['type'] == 'buy':
                        profit_long += profit
                    else:
                        profit_short += profit
                else:
                    if position['type'] == 'buy':
                        profit_second_long += profit
                    else:
                        profit_second_short += profit
                balance += profit
                position = None
                open_tp_orders = []

    # Log summary of trades for this symbol and timeframe
    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "trades_analyzed": trades_analyzed,
        "trades_executed_long": trades_executed_long,
        "trades_executed_short": trades_executed_short,
        "trades_executed_second_long": trades_executed_second_long,
        "trades_executed_second_short": trades_executed_second_short,
        "profit_long": profit_long,
        "profit_short": profit_short,
        "profit_second_long": profit_second_long,
        "profit_second_short": profit_second_short,
        "net_profit": profit_long + profit_short + profit_second_long + profit_second_short,
        "final_balance": balance
    }

    logger.info(f"Backtest complete for {symbol} on {timeframe}")
    logger.info(f"Final balance: ${balance:.2f}")

    return summary

def run_backtests():
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(config['symbols'])) as executor:
            futures = [executor.submit(backtest, symbol_config) for symbol_config in config['symbols']]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Log overall summary
        summary_logger.info("Overall Backtest Results:")
        total_net_profit = 0
        for result in results:
            if result:
                summary_logger.info(f"Summary for {result['symbol']} on {result['timeframe']}:")
                summary_logger.info(f"Trades analyzed: {result['trades_analyzed']}")
                summary_logger.info(f"Trades executed (long): {result['trades_executed_long']}")
                summary_logger.info(f"Trades executed (short): {result['trades_executed_short']}")
                summary_logger.info(f"Trades executed (second long): {result['trades_executed_second_long']}")
                summary_logger.info(f"Trades executed (second short): {result['trades_executed_second_short']}")
                summary_logger.info(f"Total profit (long): {result['profit_long']:.2f}")
                summary_logger.info(f"Total profit (short): {result['profit_short']:.2f}")
                summary_logger.info(f"Total profit (second long): {result['profit_second_long']:.2f}")
                summary_logger.info(f"Total profit (second short): {result['profit_second_short']:.2f}")
                summary_logger.info(f"Net Profit/Loss: {result['net_profit']:.2f}")
                summary_logger.info(f"Final Balance: ${result['final_balance']:.2f}")
                summary_logger.info("------------------------")
                total_net_profit += result['net_profit']

        summary_logger.info(f"Total Net Profit Across All Symbols: ${total_net_profit:.2f}")

    except KeyboardInterrupt:
        logging.info("Backtests interrupted by user")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    logging.info("Starting Wavy Tunnel Strategy Backtests")
    logging.info(f"Date range: {GLOBAL_START_DATE.date()} to {GLOBAL_END_DATE.date()}")
    run_backtests()