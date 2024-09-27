import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from config import Config
from metatrader.connection import initialize_mt5, shutdown_mt5
from strategy.tunnel_strategy import (
    check_broker_connection, check_market_open, execute_trade, place_pending_order, calculate_ema, detect_peaks_and_dips, check_entry_conditions, calculate_position_size, get_data
)
from backtesting.backtest import run_backtest
from utils.logger import setup_logging
from utils.error_handling import handle_error
from utils.mt5_log_checker import start_log_checking, stop_log_checking
import logging
import argparse
import os
import time

def clear_log_file():
    with open("app.log", "w"):
        pass

def check_auto_trading_enabled():
    """Check if global auto trading is enabled and log the status."""
    if not mt5.initialize():
        logging.error("Failed to initialize MetaTrader5 for checking auto trading status.")
        return False
    global_autotrading_enabled = mt5.terminal_info().trade_allowed
    mt5.shutdown()
    if not global_autotrading_enabled:
        logging.error("Global auto trading is disabled. Please enable it manually in the MetaTrader 5 terminal.")
    else:
        logging.info("Global auto trading is enabled.")
    return global_autotrading_enabled

def validate_mt5_and_symbol(symbol):
    if not mt5.initialize():
        logging.error("Failed to initialize MT5 connection.")
        return False
    if not mt5.symbol_select(symbol, True):
        logging.error(f"Failed to select symbol {symbol}")
        return False
    return True

def log_mt5_version():
    if mt5.initialize():
        version = mt5.version()
        logging.info(f"MetaTrader5 version: {version}")
        mt5.shutdown()
    else:
        logging.error("Failed to initialize MT5 for version check.")

def get_account_info_with_retry(max_attempts=3, delay=2):
    for attempt in range(max_attempts):
        account_info = mt5.account_info()
        if account_info is not None:
            return account_info
        logging.warning(f"Failed to get account info. Attempt {attempt + 1} of {max_attempts}.")
        time.sleep(delay)
    return None

def check_mt5_login():
    if not mt5.login(login=Config.MT5_LOGIN, password=Config.MT5_PASSWORD, server=Config.MT5_SERVER):
        error_code = mt5.last_error()
        logging.error(f"Failed to login to MT5. Error code: {error_code}")
        return False
    return True

def wait_for_mt5_terminal_load(max_wait_time=30):
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if mt5.terminal_info() is not None:
            return True
        time.sleep(1)
    return False

def run_backtest_func():
    try:
        logging.info("Initializing MetaTrader5...")
        if not initialize_mt5(Config.MT5_PATH):
            raise Exception("Failed to initialize MetaTrader5")
        logging.info("MetaTrader5 initialized successfully.")

        if not check_auto_trading_enabled():
            return

        for symbol in Config.SYMBOLS:
            if not validate_mt5_and_symbol(symbol):
                continue

            logging.info(f"Running backtest for {symbol}...")
            start_date = datetime.strptime(Config.BACKTEST_START_DATE, "%Y-%m-%d") if Config.BACKTEST_START_DATE else datetime(2023, 1, 1)
            end_date = datetime.strptime(Config.BACKTEST_END_DATE, "%Y-%m-%d") if Config.BACKTEST_END_DATE else datetime.now()
            initial_balance = 10000
            risk_percent = Config.RISK_PER_TRADE
            stop_loss_pips = 20
            pip_value = Config.PIP_VALUE

            backtest_data = get_data(symbol, mode='backtest', start_date=start_date, end_date=end_date, timeframe=Config.MT5_TIMEFRAME_VALUE)
            if backtest_data is not None and not backtest_data.empty:
                logging.info(f"Backtest data retrieved for {symbol}. Shape: {backtest_data.shape}")
                logging.info(f"Date range: from {backtest_data['time'].min()} to {backtest_data['time'].max()}")
                logging.info(f"Backtest data head:\n{backtest_data.head()}")
            else:
                logging.error(f"No historical data retrieved for {symbol} for backtesting. Timeframe: {Config.MT5_TIMEFRAME}, Start: {start_date}, End: {end_date}")
                continue

            if len(backtest_data) < 20:
                logging.error(f"Not enough data for symbol {symbol} to perform backtest")
                continue

            backtest_data.loc[:, 'close'] = pd.to_numeric(backtest_data['close'], errors='coerce')

            try:
                result = run_backtest(
                    symbol=symbol,
                    initial_balance=initial_balance,
                    risk_percent=risk_percent,
                    min_take_profit=Config.MIN_TP_PROFIT,
                    max_loss_per_day=Config.MAX_LOSS_PER_DAY,
                    starting_equity=Config.STARTING_EQUITY,
                    stop_loss_pips=stop_loss_pips,
                    pip_value=pip_value,
                    max_trades_per_day=Config.LIMIT_NO_OF_TRADES,
                    data=backtest_data,
                    start_date=start_date,
                    end_date=end_date
                )
                if result:
                    logging.info(f"Backtest results for {symbol}: {result}")
                else:
                    logging.warning(f"Backtest for {symbol} did not produce results.")
                logging.info(f"Backtest completed successfully for {symbol}.")
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

        if not wait_for_mt5_terminal_load():
            raise Exception("MetaTrader5 terminal did not load within the expected time")

        if not check_mt5_login():
            raise Exception("Failed to login to MetaTrader5")

        if not check_auto_trading_enabled():
            return

        account_info = get_account_info_with_retry()
        if account_info is None:
            error_code = mt5.last_error()
            error_desc = mt5.last_error_description()
            raise Exception(f"Failed to get account info. Error code: {error_code}, Description: {error_desc}")

        if account_info.server.endswith("demo"):
            logging.info("Trading on a demo account.")
        else:
            logging.info("Trading on a live account.")

        starting_balance = account_info.balance
        current_balance = starting_balance
        logging.info(f"Starting balance: {starting_balance:.2f}")

        if not check_broker_connection():
            return

        if not check_market_open():
            return

        daily_trades = 0
        total_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        max_drawdown_reached = False

        start_time = time.time()
        max_duration = 1 * 1800  # 30 minutes for testing, adjust as needed
        last_log_time = time.time()

        while time.time() - start_time < max_duration:
            if max_drawdown_reached:
                logging.info("Maximum drawdown reached. Stopping trading.")
                break

            if time.time() - last_log_time > 300:  # Log every 5 minutes
                logging.info("Strategy still running...")
                last_log_time = time.time()

            current_day = datetime.now().date()
            if daily_trades >= Config.LIMIT_NO_OF_TRADES:
                logging.info("Maximum number of trades for the day reached. Stopping trading for today.")
                time.sleep(86400)
                daily_trades = 0
                continue

            for symbol in Config.SYMBOLS:
                if not validate_mt5_and_symbol(symbol):
                    continue

                logging.info(f"Processing symbol: {symbol}")

                live_data = get_data(symbol, mode='live', timeframe=Config.MT5_TIMEFRAME_VALUE)
                if live_data is None or live_data.empty:
                    logging.error(f"Failed to retrieve live data for {symbol}")
                    continue

                df = live_data

                logging.info(f"Dataframe created with tick data: {df.tail()}")

                if 'last' not in df.columns:
                    df['last'] = (df['bid'] + df['ask']) / 2
                    logging.info("'last' price calculated from 'bid' and 'ask'")

                df['close'] = df['last']
                if df['close'].eq(0).any():
                    df['close'] = (df['bid'] + df['ask']) / 2

                logging.info(f"Close price values: {df['close'].tail()}")

                if 'high' not in df.columns or 'low' not in df.columns:
                    df['high'] = df['ask']
                    df['low'] = df['bid']

                df['wavy_h'] = calculate_ema(df['high'], 34)
                df['wavy_c'] = calculate_ema(df['close'], 34)
                df['wavy_l'] = calculate_ema(df['low'], 34)
                df['tunnel1'] = calculate_ema(df['close'], 144)
                df['tunnel2'] = calculate_ema(df['close'], 169)
                df['long_term_ema'] = calculate_ema(df['close'], 200)

                logging.info(f"Indicator values for {symbol}:")
                logging.info(f"Wavy H: {df['wavy_h'].iloc[-1]:.5f}")
                logging.info(f"Wavy C: {df['wavy_c'].iloc[-1]:.5f}")
                logging.info(f"Wavy L: {df['wavy_l'].iloc[-1]:.5f}")
                logging.info(f"Tunnel1: {df['tunnel1'].iloc[-1]:.5f}")
                logging.info(f"Tunnel2: {df['tunnel2'].iloc[-1]:.5f}")
                logging.info(f"Long-term EMA: {df['long_term_ema'].iloc[-1]:.5f}")

                peaks, dips = detect_peaks_and_dips(df, 21)
                logging.info(f"Number of peaks detected: {len(peaks)}")
                logging.info(f"Number of dips detected: {len(dips)}")

                buy_condition, sell_condition = check_entry_conditions(df.iloc[-1], peaks, dips, symbol)
                logging.info(f"Entry conditions for {symbol}: Buy = {buy_condition}, Sell = {sell_condition}")

                std_dev = df['close'].rolling(window=20).std().iloc[-1]
                min_sl_tp_distance = 20 * Config.PIP_VALUE  # 20 pips minimum distance

                if buy_condition or sell_condition:
                    account_info = get_account_info_with_retry()
                    if account_info is None:
                        raise Exception("Failed to get account info")

                    balance_before = account_info.balance
                    logging.info(f"Account balance before trade attempt: {balance_before}")

                    current_price = df.iloc[-1]['ask'] if buy_condition else df.iloc[-1]['bid']
                    sl_distance = max(1.5 * std_dev, min_sl_tp_distance)
                    tp_distance = max(2 * std_dev, min_sl_tp_distance)

                    stop_loss_pips = sl_distance / Config.PIP_VALUE
                    volume = calculate_position_size(
                        account_balance=balance_before,
                        risk_per_trade=Config.RISK_PER_TRADE,
                        stop_loss_pips=stop_loss_pips,
                        pip_value=Config.PIP_VALUE
                    )

                    trade_request = {
                        'symbol': symbol,
                        'volume': volume,
                        'type': mt5.ORDER_TYPE_BUY if buy_condition else mt5.ORDER_TYPE_SELL,
                        'price': current_price,
                        'sl': current_price - sl_distance if buy_condition else current_price + sl_distance,
                        'tp': current_price + tp_distance if buy_condition else current_price - tp_distance,
                        'deviation': 10,
                        'magic': 12345,
                        'comment': 'Tunnel Strategy',
                        'type_filling': mt5.ORDER_FILLING_FOK,
                        'type_time': mt5.ORDER_TIME_GTC
                    }

                    logging.info(f"Placing order with the following details: {trade_request}")

                    try:
                        result = execute_trade(trade_request)
                        logging.info(f"Order send result: {result}")

                        if result is None:
                            logging.error("mt5.order_send returned None. This may indicate a silent failure or an internal error.")
                        elif result.retcode != mt5.TRADE_RETCODE_DONE:
                            logging.error(f"Order failed with retcode: {result.retcode}")

                            if Config.ENABLE_PENDING_ORDER_FALLBACK:
                                logging.info("Attempting to place a pending order due to market order failure...")
                                pending_order_result = place_pending_order(trade_request)
                                if pending_order_result is not None:
                                    logging.info("Pending order placed successfully.")
                                else:
                                    logging.error("Failed to place pending order.")
                        else:
                            logging.info("Order placed successfully.")

                            total_profit += result.profit if hasattr(result, 'profit') else 0.0
                            current_balance += result.profit if hasattr(result, 'profit') else 0.0

                            daily_trades += 1
                            total_trades += 1
                            logging.info(f"Live trading iteration completed for {symbol}. Total trades today: {daily_trades}")
                            logging.info(f"Current Balance: {current_balance:.2f}")

                    except Exception as e:
                        logging.error(f"An error occurred while running strategy for {symbol}: {e}")

                    account_info = get_account_info_with_retry()
                    if account_info is None:
                        raise Exception("Failed to get account info")

                    balance_after = account_info.balance
                    logging.info(f"Account balance after trade attempt: {balance_after}")
                    logging.info(f"Balance change: {balance_after - balance_before}")

                else:
                    logging.info(f"No trade conditions met for {symbol}")

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

        try:
            account_info = get_account_info_with_retry()
            if account_info is None:
                logging.error("Failed to get final account info")
                ending_balance = current_balance
            else:
                ending_balance = account_info.balance

            logging.info("Summary of Trading Session:")
            logging.info(f"Total trades: {total_trades}")
            logging.info(f"Starting balance: {starting_balance:.2f}")
            logging.info(f"Ending balance: {ending_balance:.2f}")
            logging.info(f"Total profit/loss: {ending_balance - starting_balance:.2f}")
        except Exception as e:
            logging.error(f"Error in finalizing trading session: {str(e)}")

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
        log_mt5_version()

        logging.info("LOGGING ALL THE CONFIG SETTINGS")
        Config.log_config()

        start_log_checking()

        if args.ui:
            run_ui(run_backtest_func, run_live_trading_func, clear_log_file, open_log_file)
        else:
            print("Choose an option:")
            print("1. Run Backtesting")
            print("2. Run Live Trading")
            choice = input("Enter your choice (1 or 2): ")

            if choice == "1":
                logging.info("User selected Backtesting")
                run_backtest_func()
            elif choice == "2":
                logging.info("User selected Live Trading")
                run_live_trading_func()
            else:
                logging.warning(f"Invalid choice entered: {choice}")
                print("Invalid choice. Exiting...")

    except Exception as e:
        error_code = mt5.last_error()
        error_message = str(e)
        handle_error(e, f"An error occurred in the main function: {error_code} - {error_message}. Timeframe: {Config.MT5_TIMEFRAME_VALUE}")

    finally:
        stop_log_checking()

if __name__ == '__main__':
    main()