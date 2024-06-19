# import MetaTrader5 as mt5
# import pandas as pd
# from datetime import datetime
# from config import Config
# from metatrader.connection import initialize_mt5, shutdown_mt5
# from metatrader.data_retrieval import get_historical_data
# from strategy.tunnel_strategy import run_strategy, calculate_ema, detect_peaks_and_dips, check_entry_conditions
# from backtesting.backtest import run_backtest
# from utils.logger import setup_logging
# from utils.error_handling import handle_error
# import logging
# import argparse
# from ui import run_ui


# def run_backtest_func():
#     try:
#         # Initialize MetaTrader5
#         logging.info("Initializing MetaTrader5...")
#         if not initialize_mt5(Config.MT5_PATH):
#             raise Exception("Failed to initialize MetaTrader5")
#         logging.info("MetaTrader5 initialized successfully.")

#         for symbol in Config.SYMBOLS:
#             logging.info("Running backtest...")
#             start_date = datetime(2024, 6, 1)
#             end_date = datetime.now()
#             initial_balance = 10000
#             risk_percent = Config.RISK_PER_TRADE
#             stop_loss_pips = 20  # Example value for stop loss in pips
#             pip_value = Config.PIP_VALUE

#             backtest_data = get_historical_data(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
#             if backtest_data is not None and not backtest_data.empty:
#                 logging.info(f"Backtest data shape: {backtest_data.shape}")
#                 logging.info(f"Backtest data head:\n{backtest_data.head()}")
#             else:
#                 logging.error(f"No historical data retrieved for {symbol} for backtesting")
#                 continue

#             # Ensure there are enough data points for the indicators
#             if len(backtest_data) < 20:
#                 logging.error(f"Not enough data for symbol {symbol} to perform backtest")
#                 continue

#             try:
#                 run_backtest(
#                     symbol=symbol,
#                     data=backtest_data,
#                     initial_balance=initial_balance,
#                     risk_percent=risk_percent,
#                     min_take_profit=Config.MIN_TP_PROFIT,
#                     max_loss_per_day=Config.MAX_LOSS_PER_DAY,
#                     starting_equity=Config.STARTING_EQUITY,
#                     max_trades_per_day=Config.LIMIT_NO_OF_TRADES,
#                     stop_loss_pips=stop_loss_pips,  # Pass stop_loss_pips
#                     pip_value=pip_value              # Pass pip_value
#                 )
#                 logging.info("Backtest completed successfully.")
#             except Exception as e:
#                 handle_error(e, f"An error occurred during backtesting for {symbol}")

#     except Exception as e:
#         error_code = mt5.last_error()
#         error_message = "An error occurred"
#         handle_error(e, f"An error occurred in the run_backtest_func: {error_code} - {error_message}")

#     finally:
#         logging.info("Shutting down MetaTrader5...")
#         shutdown_mt5()
#         logging.info("MetaTrader5 connection gracefully shut down.")


# def run_live_trading_func():
#     try:
#         # Initialize MetaTrader5
#         logging.info("Initializing MetaTrader5...")
#         if not initialize_mt5(Config.MT5_PATH):
#             raise Exception("Failed to initialize MetaTrader5")
#         logging.info("MetaTrader5 initialized successfully.")

#         for symbol in Config.SYMBOLS:
#             logging.info("Running live trading...")
#             run_strategy(
#                 symbols=[symbol],
#                 mt5_init=mt5,
#                 timeframe=mt5.TIMEFRAME_M1,
#                 lot_size=0.01,
#                 min_take_profit=Config.MIN_TP_PROFIT,
#                 max_loss_per_day=Config.MAX_LOSS_PER_DAY,
#                 starting_equity=Config.STARTING_EQUITY,
#                 max_trades_per_day=Config.LIMIT_NO_OF_TRADES,
#                 run_backtest=False
#             )
#             logging.info("Live trading completed.")

#     except Exception as e:
#         error_code = mt5.last_error()
#         error_message = "An error occurred"
#         handle_error(e, f"An error occurred in the run_live_trading_func: {error_code} - {error_message}")

#     finally:
#         logging.info("Shutting down MetaTrader5...")
#         shutdown_mt5()
#         logging.info("MetaTrader5 connection gracefully shut down.")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ui", action="store_true", help="Run the UI")
#     args = parser.parse_args()

#     try:
#         setup_logging()
#         logging.info("STARTING APPLICATION")

#         # Log configuration settings
#         logging.info("LOGGING ALL THE CONFIG SETTINGS")
#         Config.log_config()

#         if args.ui:
#             run_ui(run_backtest_func, run_live_trading_func)
#         else:
#             # Prompt the user to choose between backtesting and live trading
#             print("Choose an option:")
#             print("1. Run Backtesting")
#             print("2. Run Live Trading")
#             choice = input("Enter your choice (1 or 2): ")

#             if choice == "1":
#                 run_backtest_func()
#             elif choice == "2":
#                 run_live_trading_func()
#             else:
#                 print("Invalid choice. Exiting...")

#     except Exception as e:
#         error_code = mt5.last_error()
#         error_message = "An error occurred"
#         handle_error(e, f"An error occurred in the main function: {error_code} - {error_message}")

# if __name__ == '__main__':
#     main()

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


def run_backtest_func():
    try:
        # Initialize MetaTrader5
        logging.info("Initializing MetaTrader5...")
        if not initialize_mt5(Config.MT5_PATH):
            raise Exception("Failed to initialize MetaTrader5")
        logging.info("MetaTrader5 initialized successfully.")

        for symbol in Config.SYMBOLS:
            logging.info("Running backtest...")
            start_date = datetime(2024, 3, 1)
            end_date = datetime.now()
            initial_balance = 10000
            risk_percent = Config.RISK_PER_TRADE
            stop_loss_pips = 20  # Example value for stop loss in pips
            pip_value = Config.PIP_VALUE

            backtest_data = get_historical_data(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
            if backtest_data is not None and not backtest_data.empty:
                logging.info(f"Backtest data shape: {backtest_data.shape}")
                logging.info(f"Backtest data head:\n{backtest_data.head()}")
            else:
                logging.error(f"No historical data retrieved for {symbol} for backtesting")
                continue

            # Ensure there are enough data points for the indicators
            if len(backtest_data) < 20:
                logging.error(f"Not enough data for symbol {symbol} to perform backtest")
                continue

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
                    stop_loss_pips=stop_loss_pips,  # Pass stop_loss_pips
                    pip_value=pip_value              # Pass pip_value
                )
                logging.info("Backtest completed successfully.")
            except Exception as e:
                handle_error(e, f"An error occurred during backtesting for {symbol}")

    except Exception as e:
        error_code = mt5.last_error()
        error_message = "An error occurred"
        handle_error(e, f"An error occurred in the run_backtest_func: {error_code} - {error_message}")

    finally:
        logging.info("Shutting down MetaTrader5...")
        shutdown_mt5()
        logging.info("MetaTrader5 connection gracefully shut down.")


def run_live_trading_func():
    try:
        # Initialize MetaTrader5
        logging.info("Initializing MetaTrader5...")
        if not initialize_mt5(Config.MT5_PATH):
            raise Exception("Failed to initialize MetaTrader5")
        logging.info("MetaTrader5 initialized successfully.")

        for symbol in Config.SYMBOLS:
            logging.info("Running live trading...")
            run_strategy(
                symbols=[symbol],
                mt5_init=mt5,
                timeframe=mt5.TIMEFRAME_M1,
                lot_size=0.01,
                min_take_profit=Config.MIN_TP_PROFIT,
                max_loss_per_day=Config.MAX_LOSS_PER_DAY,
                starting_equity=Config.STARTING_EQUITY,
                max_trades_per_day=Config.LIMIT_NO_OF_TRADES,
                run_backtest=False
            )
            logging.info("Live trading completed.")

    except Exception as e:
        error_code = mt5.last_error()
        error_message = "An error occurred"
        handle_error(e, f"An error occurred in the run_live_trading_func: {error_code} - {error_message}")

    finally:
        logging.info("Shutting down MetaTrader5...")
        shutdown_mt5()
        logging.info("MetaTrader5 connection gracefully shut down.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", action="store_true", help="Run the UI")
    args = parser.parse_args()

    try:
        setup_logging()
        logging.info("STARTING APPLICATION")

        # Log configuration settings
        logging.info("LOGGING ALL THE CONFIG SETTINGS")
        Config.log_config()

        if args.ui:
            run_ui(run_backtest_func, run_live_trading_func)
        else:
            # Prompt the user to choose between backtesting and live trading
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
        error_message = "An error occurred"
        handle_error(e, f"An error occurred in the main function: {error_code} - {error_message}")

if __name__ == '__main__':
    main()
