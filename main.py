import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from config import Config
from metatrader.connection import initialize_mt5, shutdown_mt5
from metatrader.data_retrieval import get_historical_data
from strategy.tunnel_strategy import run_strategy, calculate_ema, detect_peaks_and_dips, check_entry_conditions
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
