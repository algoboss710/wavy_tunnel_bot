import MetaTrader5 as mt5
from config import Config
from metatrader.connection import initialize_mt5, shutdown_mt5
from metatrader.data_retrieval import get_historical_data
from strategy.tunnel_strategy import run_strategy
from backtesting.backtest import run_backtest
from utils.logger import setup_logging
from utils.error_handling import handle_error

def main():
    try:
        # Set up logging
        setup_logging()

        # Initialize MetaTrader5
        if not initialize_mt5(Config.MT5_PATH):
            raise Exception("Failed to initialize MetaTrader5")

        # Run the trading strategy
        run_strategy(
            symbols=Config.SYMBOLS,
            mt5_init=mt5,
            timeframe=mt5.TIMEFRAME_M1,
            lot_size=0.01,
            min_take_profit=Config.MIN_TP_PROFIT,
            max_loss_per_day=Config.MAX_LOSS_PER_DAY,
            starting_equity=Config.STARTING_EQUITY,
            max_trades_per_day=Config.LIMIT_NO_OF_TRADES
        )

        # Run backtesting
        symbol = 'EURUSD'
        start_date = '2022-01-01'
        end_date = '2022-12-31'
        timeframe = mt5.TIMEFRAME_H1
        initial_balance = 10000
        risk_percent = 0.02

        historical_data = get_historical_data(symbol, timeframe, start_date, end_date)
        if historical_data is None or len(historical_data) == 0:
            raise ValueError(f"No historical data retrieved for {symbol}")
    
        run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            initial_balance=initial_balance,
            risk_percent=risk_percent,
            min_take_profit=Config.MIN_TP_PROFIT,
            max_loss_per_day=Config.MAX_LOSS_PER_DAY,
            starting_equity=Config.STARTING_EQUITY,
            max_trades_per_day=Config.LIMIT_NO_OF_TRADES
        )

    except Exception as e:
        handle_error(e, "An error occurred in the main function")

    finally:
        # Shutdown MetaTrader5
        shutdown_mt5()

if __name__ == '__main__':
    main()



# import signal
# import sys
# from scheduler import setup_schedule, run_scheduled_tasks
# from utils.error_handling import handle_error, critical_error
# from strategy.trade_logic import calculate_position_size, entry_long, entry_short, exit_trade
# from utils.data_validation import validate_data, sanitize_data
# from config import Config
# from metatrader.connection import initialize_mt5
# from metatrader.trade_management import get_open_positions, should_exit_position, generate_trade_signal
# from metatrader.connection import initialize_mt5, get_account_info


# def signal_handler(signum, frame):
#     critical_error("Signal received, shutting down", f"Signal handler triggered with signal: {signum}")
#     sys.exit(0)

# # Register the signal handler
# signal.signal(signal.SIGINT, signal_handler)

# def main():
#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)

#     try:
#         setup_schedule()
        
#         if initialize_mt5():
#             account_info = get_account_info()
#             account_balance = account_info["balance"]

#             for symbol in Config.SYMBOLS:
#                 data = get_historical_data(symbol, Config.MT5_TIMEFRAME, start_time, end_time)
#                 sanitized_data = sanitize_data(data)

#                 if validate_data(sanitized_data, TradeRequestSchema.schema()):
#                     signal = generate_trade_signal(sanitized_data, period, deviation)
#                     price = sanitized_data["close"].iloc[-1]
#                     stop_loss_pips = 20
#                     take_profit_pips = 40
#                     deviation = 10
#                     magic = 12345
#                     comment = "Tunnel Strategy"

#                     position_size = calculate_position_size(account_balance, Config.RISK_PER_TRADE, stop_loss_pips, Config.PIP_VALUE)

#                     if signal == 'BUY':
#                         sl = price - (stop_loss_pips * Config.PIP_VALUE)
#                         tp = price + (take_profit_pips * Config.PIP_VALUE)
#                         entry_long(symbol, position_size, price, sl, tp, deviation, magic, comment)
#                     elif signal == 'SELL':
#                         sl = price + (stop_loss_pips * Config.PIP_VALUE)
#                         tp = price - (take_profit_pips * Config.PIP_VALUE)
#                         entry_short(symbol, position_size, price, sl, tp, deviation, magic, comment)
#                 else:
#                     logging.error("Invalid input data")

#             open_positions = get_open_positions()
#             for position in open_positions:
#                 if should_exit_position(position):
#                     exit_trade(position.ticket)

#         run_scheduled_tasks()

#     except Exception as e:
#         handle_error(e, "Failed to execute trading logic or validate/sanitize input data")

# if __name__ == "__main__":
#     main()