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
import MetaTrader5 as mt5
print(mt5.__version__)
import logging

def main():
    try:
        setup_logging()
        logging.info("Starting the main function.")

        # Initialize MetaTrader5
        logging.info("Initializing MetaTrader5...")
        if not initialize_mt5(Config.MT5_PATH):
            raise Exception("Failed to initialize MetaTrader5")
        logging.info("MetaTrader5 initialized successfully.")

        for symbol in Config.SYMBOLS:
            # Check symbol availability
            logging.info(f"Selecting symbol {symbol}...")
            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"Symbol {symbol} not available")
            logging.info(f"Symbol {symbol} selected successfully.")

            # Retrieve historical price data
            logging.info(f"Retrieving historical price data for {symbol}...")
            historical_data = get_historical_data(symbol, mt5.TIMEFRAME_H1, datetime.now() - pd.Timedelta(days=30), datetime.now())
            if historical_data is None or historical_data.empty:
                raise ValueError(f"No historical data retrieved for {symbol}")
            print(f"Historical data shape after retrieval: {historical_data.shape}")
            print(f"Historical data head after retrieval:\n{historical_data.head()}")
            logging.info("Historical price data retrieved successfully.")
            logging.info(historical_data.head())

            # Calculate indicators
            logging.info("Calculating Wavy Tunnel indicator...")
            historical_data['wavy_h'] = calculate_ema(historical_data['high'], 34)
            historical_data['wavy_c'] = calculate_ema(historical_data['close'], 34)
            historical_data['wavy_l'] = calculate_ema(historical_data['low'], 34)
            historical_data['tunnel1'] = calculate_ema(historical_data['close'], 144)
            historical_data['tunnel2'] = calculate_ema(historical_data['close'], 169)
            historical_data['long_term_ema'] = calculate_ema(historical_data['close'], 200)
            logging.info("Indicators calculated.")

            # Detect peaks and dips
            logging.info("Detecting peaks and dips...")
            peaks, dips = detect_peaks_and_dips(historical_data, peak_type=21)
            logging.info(f"Peaks: {peaks[:5]}")
            logging.info(f"Dips: {dips[:5]}")

            # Generate entry signals
            logging.info("Generating entry signals...")
            historical_data['buy_signal'], historical_data['sell_signal'] = zip(*historical_data.apply(lambda x: check_entry_conditions(x, peaks, dips, symbol), axis=1))
            logging.info("Entry signals generated.")
            logging.info(historical_data[['time', 'close', 'buy_signal', 'sell_signal']].head())

            # Run backtesting
            logging.info("Running backtest...")
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2022, 12, 31)
            initial_balance = 10000
            risk_percent = 0.02

            backtest_data = get_historical_data(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
            if backtest_data is None or backtest_data.empty:
                logging.error(f"No historical data retrieved for {symbol} for backtesting")
            else:
                logging.info("Backtest data retrieved successfully.")
            run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=mt5.TIMEFRAME_H1,
                initial_balance=initial_balance,
                risk_percent=risk_percent,
                min_take_profit=Config.MIN_TP_PROFIT,
                max_loss_per_day=Config.MAX_LOSS_PER_DAY,
                starting_equity=Config.STARTING_EQUITY,
                max_trades_per_day=Config.LIMIT_NO_OF_TRADES
                )
            logging.info("Backtest completed successfully.")
            # # Run trading strategy
            # logging.info("Running trading strategy...")
            # run_strategy(
            #     symbols=[symbol],
            #     mt5_init=mt5,
            #     timeframe=mt5.TIMEFRAME_M1,
            #     lot_size=0.01,
            #     min_take_profit=Config.MIN_TP_PROFIT,
            #     max_loss_per_day=Config.MAX_LOSS_PER_DAY,
            #     starting_equity=Config.STARTING_EQUITY,
            #     max_trades_per_day=Config.LIMIT_NO_OF_TRADES
            # )
            # print(f"Historical data shape after running strategy: {historical_data.shape}")
            # print(f"Historical data head after running strategy:\n{historical_data.head()}")
            # logging.info("Trading strategy completed.")

            # # Run backtesting
            # logging.info("Running backtest...")
            # start_date = datetime(2022, 1, 1)
            # end_date = datetime(2022, 12, 31)
            # initial_balance = 10000
            # risk_percent = 0.02

            # backtest_data = get_historical_data(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
            # if backtest_data is None or backtest_data.empty:
            #     raise ValueError(f"No historical data retrieved for {symbol} for backtesting")
            # logging.info("Backtest data retrieved successfully.")

            # run_backtest(
            #     symbol=symbol,
            #     start_date=start_date,
            #     end_date=end_date,
            #     timeframe=mt5.TIMEFRAME_H1,
            #     initial_balance=initial_balance,
            #     risk_percent=risk_percent,
            #     min_take_profit=Config.MIN_TP_PROFIT,
            #     max_loss_per_day=Config.MAX_LOSS_PER_DAY,
            #     starting_equity=Config.STARTING_EQUITY,
            #     max_trades_per_day=Config.LIMIT_NO_OF_TRADES
            # )
            # logging.info("Backtest completed successfully.")

    except Exception as e:
        error_code = mt5.last_error()
        error_message = "An error occurred"
        handle_error(e, f"An error occurred in the main function: {error_code} - {error_message}")

    finally:
        logging.info("Shutting down MetaTrader5...")
        shutdown_mt5()
        logging.info("MetaTrader5 connection gracefully shut down.")

if __name__ == '__main__':
    main()

# import MetaTrader5 as mt5
# from datetime import datetime
# from config import Config
# from metatrader.connection import initialize_mt5, shutdown_mt5
# from metatrader.data_retrieval import get_historical_data
# from strategy.tunnel_strategy import run_strategy
# from backtesting.backtest import run_backtest
# from utils.logger import setup_logging
# from utils.error_handling import handle_error
# import logging

# def main():
#     try:
#         # Set up logging
#         setup_logging()
#         logging.info("Logging setup complete.")

#         # Initialize MetaTrader5
#         logging.info("Initializing MetaTrader5...")
#         if not initialize_mt5(Config.MT5_PATH):
#             raise Exception("Failed to initialize MetaTrader5")
#         logging.info("MetaTrader5 initialized successfully.")

#         # Check symbol availability
#         symbol = 'EURUSD'
#         logging.info(f"Selecting symbol {symbol}...")
#         if not mt5.symbol_select(symbol, True):
#             raise ValueError(f"Symbol {symbol} not available")
#         logging.info(f"Symbol {symbol} selected successfully.")

#         # Run the trading strategy
#         logging.info("Running trading strategy...")
#         run_strategy(
#             symbols=Config.SYMBOLS,
#             mt5_init=mt5,
#             timeframe=mt5.TIMEFRAME_M1,  # Correct timeframe
#             lot_size=0.01,
#             min_take_profit=Config.MIN_TP_PROFIT,
#             max_loss_per_day=Config.MAX_LOSS_PER_DAY,
#             starting_equity=Config.STARTING_EQUITY,
#             max_trades_per_day=Config.LIMIT_NO_OF_TRADES
#         )
#         logging.info("Trading strategy completed.")

#         # Run backtesting
#         logging.info("Running backtest...")
#         start_date = datetime(2022, 1, 1)
#         end_date = datetime(2022, 12, 31)
#         timeframe = mt5.TIMEFRAME_H1  # Correct timeframe
#         initial_balance = 10000
#         risk_percent = 0.02

#         historical_data = get_historical_data(symbol, timeframe, start_date, end_date)
#         if historical_data is None or len(historical_data) == 0:
#             raise ValueError(f"No historical data retrieved for {symbol}")
#         logging.info("Historical data retrieved successfully.")
    
#         run_backtest(
#             symbol=symbol,
#             start_date=start_date,
#             end_date=end_date,
#             timeframe=timeframe,
#             initial_balance=initial_balance,
#             risk_percent=risk_percent,
#             min_take_profit=Config.MIN_TP_PROFIT,
#             max_loss_per_day=Config.MAX_LOSS_PER_DAY,
#             starting_equity=Config.STARTING_EQUITY,
#             max_trades_per_day=Config.LIMIT_NO_OF_TRADES
#         )
#         logging.info("Backtest completed successfully.")

#     except Exception as e:
#         handle_error(e, "An error occurred in the main function")

#     finally:
#         # Shutdown MetaTrader5
#         logging.info("Shutting down MetaTrader5...")
#         shutdown_mt5()
#         logging.info("MetaTrader5 connection gracefully shut down.")

# if __name__ == '__main__':
#     main()
