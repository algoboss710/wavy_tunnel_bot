import schedule
import time
import logging
from strategy.tunnel_strategy import run_strategy
from metatrader.connection import initialize_mt5
from config import Config
from utils.error_handling import handle_error, warn_error
from strategy.trade_logic import calculate_position_size, entry_long, entry_short, exit_trade
from utils.data_validation import validate_data, sanitize_data
from metatrader.trade_management import get_open_positions, should_exit_position

def initialize_strategy():
    try:
        logging.info("Initializing strategy on server: %s", Config.MT5_SERVER)
        mt5_init = initialize_mt5(
            login=Config.MT5_LOGIN,
            password=Config.MT5_PASSWORD,
            server=Config.MT5_SERVER,
            path=Config.MT5_PATH,
        )

        run_strategy(
            symbols=Config.SYMBOLS,
            mt5_init=mt5_init,
            timeframe=Config.MT5_TIMEFRAME,
            lot_size=Config.MT5_LOT_SIZE,
            min_take_profit=Config.MIN_TP_PROFIT,
            max_loss_per_day=Config.MAX_LOSS_PER_DAY,
            starting_equity=Config.STARTING_EQUITY,
            max_traders_per_day=Config.LIMIT_NO_OF_TRADES
        )
    except Exception as e:
        warn_error(e, "Error initializing strategy")

def run_scheduled_tasks():
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            handle_error(e, "Error during scheduled task execution")

def run_trading_task():
    try:
        account_info = get_account_info()
        account_balance = account_info["balance"]

        for symbol in Config.SYMBOLS:
            data = get_historical_data(symbol, Config.MT5_TIMEFRAME, start_time, end_time)
            sanitized_data = sanitize_data(data)

            if validate_data(sanitized_data, TradeRequestSchema.schema()):
                signal = generate_trade_signal(sanitized_data, period, deviation)
                price = sanitized_data["close"].iloc[-1]
                stop_loss_pips = 20
                take_profit_pips = 40
                deviation = 10
                magic = 12345
                comment = "Tunnel Strategy"

                position_size = calculate_position_size(account_balance, Config.RISK_PER_TRADE, stop_loss_pips, Config.PIP_VALUE)

                if signal == 'BUY':
                    sl = price - (stop_loss_pips * Config.PIP_VALUE)
                    tp = price + (take_profit_pips * Config.PIP_VALUE)
                    entry_long(symbol, position_size, price, sl, tp, deviation, magic, comment)
                elif signal == 'SELL':
                    sl = price + (stop_loss_pips * Config.PIP_VALUE)
                    tp = price - (take_profit_pips * Config.PIP_VALUE)
                    entry_short(symbol, position_size, price, sl, tp, deviation, magic, comment)
            else:
                logging.error("Invalid input data")

        open_positions = get_open_positions()
        for position in open_positions:
            if should_exit_position(position):
                exit_trade(position.ticket)

    except Exception as e:
        handle_error(e, "Failed to execute trading task or validate/sanitize input data")

def setup_schedule():
    schedule.every().day.at("09:00").do(initialize_strategy)
    schedule.every(15).minutes.do(run_trading_task)
    logging.info("Scheduler setup complete. Next run at: %s", schedule.next_run())

def adjust_schedule(market_conditions):
    if market_conditions == 'volatile':
        schedule.every(5).minutes.do(run_trading_task)
    elif market_conditions == 'calm':
        schedule.every(30).minutes.do(run_trading_task)

# Call adjust_schedule() based on market conditions
market_conditions = analyze_market_conditions()
adjust_schedule(market_conditions)