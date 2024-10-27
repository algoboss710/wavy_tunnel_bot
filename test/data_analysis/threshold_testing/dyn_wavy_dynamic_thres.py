import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import logging
import os
import asyncio
import concurrent.futures
import signal
import sys
import time

# Set up logging
log_dir = 'dy_wavy_tunnel_logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=f'{log_dir}/optimization.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# HTML logging setup
def setup_html_logger(symbol, timeframe):
    logger = logging.getLogger(f'{symbol}_{timeframe}')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{log_dir}/{symbol}_{timeframe}_results.html', mode='w')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    return logger

def log_html_header(logger, symbol, timeframe):
    logger.info(f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
    <h1>Top 10 Parameter Combinations for {symbol} on {timeframe}</h1>
    """)

def log_html_results(logger, results):
    logger.info("<table>")
    logger.info("<tr><th>Rank</th><th>Wavy Period</th><th>Tunnel Period 1</th><th>Tunnel Period 2</th><th>ATR Period</th><th>ATR Multiplier</th><th>Sharpe Ratio</th><th>Final Balance</th><th>Analyzed Trades</th><th>Executed Trades</th><th>Win Rate</th><th>Profit Factor</th></tr>")
    for rank, row in enumerate(results.itertuples(), 1):
        logger.info(f"<tr><td>{rank}</td><td>{row.wavy_period}</td><td>{row.tunnel_period1}</td><td>{row.tunnel_period2}</td><td>{row.atr_period}</td><td>{row.atr_multiplier}</td><td>{row.sharpe_ratio:.4f}</td><td>${row.final_balance:.2f}</td><td>{row.analyzed_trades}</td><td>{row.total_trades}</td><td>{row.win_rate:.2%}</td><td>{row.profit_factor:.2f}</td></tr>")
    logger.info("</table>")

def log_html_footer(logger):
    logger.info("</body></html>")

def initialize_mt5():
    if not mt5.initialize():
        logging.error("MetaTrader 5 initialization failed")
        mt5.shutdown()
        return False
    logging.info("MetaTrader 5 initialized successfully")
    return True

def get_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        logging.error(f"Failed to retrieve data for {symbol} on {timeframe} from {start_date} to {end_date}")
        return None

    df = pd.DataFrame(rates)

    if 'time' not in df.columns:
        if 'date' in df.columns:
            df['time'] = pd.to_datetime(df['date'], unit='s')
        else:
            logging.warning(f"No 'time' or 'date' column found for {symbol} on {timeframe}. Creating a dummy time index.")
            df['time'] = pd.date_range(start=start_date, periods=len(df), freq='H')
    else:
        df['time'] = pd.to_datetime(df['time'], unit='s')

    logging.info(f"Retrieved {len(df)} data points for {symbol} on {timeframe} from {start_date} to {end_date}")
    return df

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(high, low, close, period):
    tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
    return tr.rolling(window=period).mean()

def wavy_tunnel_strategy(df, wavy_period, tunnel_period1, tunnel_period2, atr_period, atr_multiplier):
    df['wavy_h'] = calculate_ema(df['high'], wavy_period)
    df['wavy_c'] = calculate_ema(df['close'], wavy_period)
    df['wavy_l'] = calculate_ema(df['low'], wavy_period)
    df['tunnel1'] = calculate_ema(df['close'], tunnel_period1)
    df['tunnel2'] = calculate_ema(df['close'], tunnel_period2)

    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    df['threshold'] = df['atr'] * atr_multiplier

    df['long_condition'] = (df['open'] > df[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1) + df['threshold']) & \
                           (df[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1) > df[['tunnel1', 'tunnel2']].max(axis=1))

    df['short_condition'] = (df['open'] < df[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1) - df['threshold']) & \
                            (df[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1) < df[['tunnel1', 'tunnel2']].min(axis=1))

    df['analyzed_trade'] = df['long_condition'] | df['short_condition']

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
            elif df['short_condition'].iloc[i]:
                position = {'type': 'short', 'entry_price': df['open'].iloc[i], 'entry_time': df['time'].iloc[i]}
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
                position = None

    return balance, trades, analyzed_trades

def optimize_parameters(df, param_ranges):
    results = []
    for wavy_period, tunnel_period1, tunnel_period2, atr_period, atr_multiplier in product(
        param_ranges['wavy_period'], param_ranges['tunnel_period1'], param_ranges['tunnel_period2'],
        param_ranges['atr_period'], param_ranges['atr_multiplier']):

        df_strategy = wavy_tunnel_strategy(df.copy(), wavy_period, tunnel_period1, tunnel_period2, atr_period, atr_multiplier)
        final_balance, trades, analyzed_trades = backtest(df_strategy)

        if trades:
            total_trades = len(trades)
            win_rate = sum(1 for trade in trades if trade['pnl'] > 0) / total_trades

            total_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
            total_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))

            epsilon = 1e-10  # Small value to avoid division by zero due to floating-point precision
            if total_loss <= epsilon:
                profit_factor = float('inf') if total_profit > epsilon else 0
            else:
                profit_factor = total_profit / total_loss

            returns = [trade['pnl'] for trade in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        else:
            total_trades = win_rate = profit_factor = sharpe_ratio = 0

        results.append({
            'wavy_period': wavy_period,
            'tunnel_period1': tunnel_period1,
            'tunnel_period2': tunnel_period2,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'final_balance': final_balance,
            'total_trades': total_trades,
            'analyzed_trades': analyzed_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio
        })

    return pd.DataFrame(results)

async def analyze_pair_timeframe(symbol, timeframe, start_date, end_date, param_ranges):
    try:
        df = await asyncio.to_thread(get_data, symbol, timeframe, start_date, end_date)
        if df is None or len(df) == 0:
            logging.error(f"No data available for {symbol} on {timeframe}. Skipping analysis.")
            return symbol, timeframe, None

        results = await asyncio.to_thread(optimize_parameters, df, param_ranges)
        if results.empty:
            logging.warning(f"No valid results for {symbol} on {timeframe}. Skipping analysis.")
            return symbol, timeframe, None

        top_results = results.sort_values('sharpe_ratio', ascending=False).head(10)

        # Log results to HTML file
        html_logger = setup_html_logger(symbol, timeframe)
        log_html_header(html_logger, symbol, timeframe)
        log_html_results(html_logger, top_results)
        log_html_footer(html_logger)

        logging.info(f"Results for {symbol} on {timeframe} logged to {log_dir}/{symbol}_{timeframe}_results.html")

        return symbol, timeframe, top_results
    except Exception as e:
        logging.error(f"Error analyzing {symbol} on {timeframe}: {str(e)}")
        return symbol, timeframe, None

def get_symbol_selection():
    all_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
    print("\nAvailable symbols:")
    for i, symbol in enumerate(all_symbols, 1):
        print(f"{i}. {symbol}")
    print("7. All symbols")
    print("8. Custom selection")

    while True:
        choice = input("\nEnter your choice (1-8): ")
        if choice == '7':
            return all_symbols
        elif choice == '8':
            custom_symbols = input("Enter symbols separated by commas (e.g., EURUSD,GBPUSD): ").split(',')
            return [symbol.strip().upper() for symbol in custom_symbols]
        elif choice.isdigit() and 1 <= int(choice) <= 6:
            return [all_symbols[int(choice) - 1]]
        else:
            print("Invalid choice. Please try again.")
async def main():
    try:
        if not initialize_mt5():
            return

        selected_symbols = get_symbol_selection()
        print(f"Selected symbols: {', '.join(selected_symbols)}")
        #, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1
        timeframes = [mt5.TIMEFRAME_D1]
        start_date = datetime(2023, 9, 1)
        end_date = datetime(2024, 10, 20)

        param_ranges = {
            'wavy_period': range(20, 51, 5),
            'tunnel_period1': range(100, 301, 20),
            'tunnel_period2': range(120, 321, 20),
            'atr_period': range(5, 31, 5),
            'atr_multiplier': np.arange(0.5, 3.1, 0.5)
        }

        tasks = []
        for symbol in selected_symbols:
            for timeframe in timeframes:
                task = analyze_pair_timeframe(symbol, timeframe, start_date, end_date, param_ranges)
                tasks.append(task)

        total_tasks = len(tasks)
        completed_tasks = 0
        start_time = time.time()

        async def status_update():
            nonlocal completed_tasks
            while completed_tasks < total_tasks:
                elapsed_time = time.time() - start_time
                print(f"Status update: {completed_tasks}/{total_tasks} tasks completed. "
                      f"Elapsed time: {elapsed_time:.2f} seconds")
                await asyncio.sleep(300)  # Update every 5 minutes

        status_task = asyncio.create_task(status_update())

        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed_tasks += 1

        status_task.cancel()

        for symbol, timeframe, top_results in results:
            if isinstance(top_results, Exception):
                print(f"Error occurred for {symbol} on {timeframe}: {str(top_results)}")
            elif top_results is not None:
                print(f"\nResults for {symbol} on {timeframe} logged to {log_dir}/{symbol}_{timeframe}_results.html")

                best_combination = top_results.iloc[0]
                print(f"\nBest parameters for {symbol} on {timeframe}:")
                print(f"Wavy Period: {best_combination['wavy_period']}")
                print(f"Tunnel Period 1: {best_combination['tunnel_period1']}")
                print(f"Tunnel Period 2: {best_combination['tunnel_period2']}")
                print(f"ATR Period: {best_combination['atr_period']}")
                print(f"ATR Multiplier: {best_combination['atr_multiplier']}")
                print(f"Sharpe Ratio: {best_combination['sharpe_ratio']:.4f}")
                print(f"Final Balance: ${best_combination['final_balance']:.2f}")
                print(f"Analyzed Trades: {best_combination['analyzed_trades']}")
                print(f"Executed Trades: {best_combination['total_trades']}")
                print(f"Win Rate: {best_combination['win_rate']:.2%}")
                print(f"Profit Factor: {best_combination['profit_factor']:.2f}")
            else:
                print(f"\nNo results available for {symbol} on {timeframe}")

        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")

    except asyncio.CancelledError:
        print("Optimization process was cancelled.")
    finally:
        mt5.shutdown()
        print("MetaTrader 5 connection closed.")

def signal_handler(signum, frame):
    raise KeyboardInterrupt()

if __name__ == "__main__":
    # Set up SIGINT (Ctrl+C) handler for all platforms
    signal.signal(signal.SIGINT, signal_handler)

    # Set up SIGTSTP (Ctrl+Z) handler only for Unix-like systems
    if sys.platform != "win32":
        def sigtstp_handler(signum, frame):
            print("\nCtrl+Z detected. Please use Ctrl+C to exit the script.")
        signal.signal(signal.SIGTSTP, sigtstp_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        if mt5.initialize():
            mt5.shutdown()
            print("MetaTrader 5 connection closed.")
        print("Script execution completed.")