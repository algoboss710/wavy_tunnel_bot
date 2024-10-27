import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import logging
import os

# Set up logging
log_dir = 'wavy_tunnel_logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=f'{log_dir}/optimization.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_mt5():
    if not mt5.initialize():
        logging.error("MetaTrader 5 initialization failed")
        mt5.shutdown()
        return False
    logging.info("MetaTrader 5 initialized successfully")
    return True

def get_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    logging.info(f"Retrieved {len(df)} data points for {symbol} from {start_date} to {end_date}")
    return df

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(high, low, close, period):
    tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
    return tr.rolling(window=period).mean()

def wavy_tunnel_strategy(df, atr_period, atr_multiplier):
    df['wavy_h'] = calculate_ema(df['high'], 34)
    df['wavy_c'] = calculate_ema(df['close'], 34)
    df['wavy_l'] = calculate_ema(df['low'], 34)
    df['tunnel1'] = calculate_ema(df['close'], 144)
    df['tunnel2'] = calculate_ema(df['close'], 169)
    df['ema_12'] = calculate_ema(df['close'], 12)

    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    df['threshold'] = df['atr'] * atr_multiplier

    df['long_condition'] = (df['open'] > df[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1) + df['threshold']) & \
                           (df[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1) > df[['tunnel1', 'tunnel2']].max(axis=1))

    df['short_condition'] = (df['open'] < df[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1) - df['threshold']) & \
                            (df[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1) < df[['tunnel1', 'tunnel2']].min(axis=1))

    df['analyzed_trade'] = df['long_condition'] | df['short_condition']

    logging.info(f"Strategy applied with ATR period: {atr_period}, ATR multiplier: {atr_multiplier}")
    return df

def backtest(df, initial_balance=10000, lot_size=0.01):
    balance = initial_balance
    position = None
    trades = []
    analyzed_trades = df['analyzed_trade'].sum()

    for i in range(1, len(df)):
        if position is None:
            if df['long_condition'].iloc[i]:
                position = {'type': 'long', 'entry_price': df['open'].iloc[i], 'entry_time': df['time'].iloc[i]}
                logging.debug(f"Entered long position at {position['entry_price']} on {position['entry_time']}")
            elif df['short_condition'].iloc[i]:
                position = {'type': 'short', 'entry_price': df['open'].iloc[i], 'entry_time': df['time'].iloc[i]}
                logging.debug(f"Entered short position at {position['entry_price']} on {position['entry_time']}")
        else:
            exit_condition = (position['type'] == 'long' and df['close'].iloc[i] < df['wavy_l'].iloc[i]) or \
                             (position['type'] == 'short' and df['close'].iloc[i] > df['wavy_h'].iloc[i])

            if exit_condition:
                exit_price = df['close'].iloc[i]
                pnl = (exit_price - position['entry_price']) * lot_size * 100000 if position['type'] == 'long' else \
                      (position['entry_price'] - exit_price) * lot_size * 100000
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': df['time'].iloc[i],
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                logging.debug(f"Closed {position['type']} position at {exit_price} on {df['time'].iloc[i]}. PnL: {pnl}")
                position = None

    logging.info(f"Backtest completed. Final balance: {balance}, Total trades: {len(trades)}, Analyzed trades: {analyzed_trades}")
    return balance, trades, analyzed_trades

def optimize_atr_parameters(symbol, timeframe, start_date, end_date, atr_param_ranges):
    df = get_data(symbol, timeframe, start_date, end_date)

    results = []
    for atr_period, atr_multiplier in product(atr_param_ranges['atr_period'], atr_param_ranges['atr_multiplier']):
        df_strategy = wavy_tunnel_strategy(df.copy(), atr_period, atr_multiplier)
        final_balance, trades, analyzed_trades = backtest(df_strategy)

        if trades:
            total_trades = len(trades)
            win_rate = sum(1 for trade in trades if trade['pnl'] > 0) / total_trades
            profit_factor = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0) / abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
            sharpe_ratio = np.mean([trade['pnl'] for trade in trades]) / np.std([trade['pnl'] for trade in trades]) if np.std([trade['pnl'] for trade in trades]) != 0 else 0
        else:
            total_trades = win_rate = profit_factor = sharpe_ratio = 0

        results.append({
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'final_balance': final_balance,
            'total_trades': total_trades,
            'analyzed_trades': analyzed_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio
        })
        logging.info(f"Optimization result for ATR period: {atr_period}, ATR multiplier: {atr_multiplier}, "
                     f"Final balance: {final_balance}, Total trades: {total_trades}, Analyzed trades: {analyzed_trades}, Sharpe ratio: {sharpe_ratio}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    if not initialize_mt5():
        exit()

    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    logging.info(f"Starting optimization for {symbol} on {mt5.TIMEFRAME_H1} timeframe from {start_date} to {end_date}")

    atr_param_ranges = {
        'atr_period': range(5, 31, 5),  # 5, 10, 15, 20, 25, 30
        'atr_multiplier': np.arange(0.5, 3.1, 0.5)  # 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    }

    results = optimize_atr_parameters(symbol, timeframe, start_date, end_date, atr_param_ranges)

    # Sort results by Sharpe ratio and print top 10
    top_results = results.sort_values('sharpe_ratio', ascending=False).head(10)
    logging.info("Top 10 ATR parameter combinations by Sharpe ratio:")
    logging.info(top_results.to_string())
    print("Top 10 ATR parameter combinations by Sharpe ratio:")
    print(top_results)

    # Print the best combination
    best_combination = top_results.iloc[0]
    logging.info("\nBest ATR parameters:")
    logging.info(f"ATR Period: {best_combination['atr_period']}")
    logging.info(f"ATR Multiplier: {best_combination['atr_multiplier']}")
    logging.info(f"Sharpe Ratio: {best_combination['sharpe_ratio']:.4f}")
    logging.info(f"Final Balance: ${best_combination['final_balance']:.2f}")
    logging.info(f"Analyzed Trades: {best_combination['analyzed_trades']}")
    logging.info(f"Executed Trades: {best_combination['total_trades']}")
    logging.info(f"Win Rate: {best_combination['win_rate']:.2%}")
    logging.info(f"Profit Factor: {best_combination['profit_factor']:.2f}")

    print("\nBest ATR parameters:")
    print(f"ATR Period: {best_combination['atr_period']}")
    print(f"ATR Multiplier: {best_combination['atr_multiplier']}")
    print(f"Sharpe Ratio: {best_combination['sharpe_ratio']:.4f}")
    print(f"Final Balance: ${best_combination['final_balance']:.2f}")
    print(f"Analyzed Trades: {best_combination['analyzed_trades']}")
    print(f"Executed Trades: {best_combination['total_trades']}")
    print(f"Win Rate: {best_combination['win_rate']:.2%}")
    print(f"Profit Factor: {best_combination['profit_factor']:.2f}")

    logging.info("Optimization completed")
    mt5.shutdown()