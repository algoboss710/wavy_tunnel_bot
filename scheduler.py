import schedule
import time
import logging
from strategy import tunnel_strategy
from metatrader import mt5_lib
from config import Config
from utils.error_handling import handle_error, warn_error

def initialize_strategy():
    try:
        logging.info("Initializing strategy on server: %s", Config.MT5_SERVER)
        mt5_init = mt5_lib.Mt5(
            login=Config.MT5_LOGIN,
            password=Config.MT5_PASSWORD,
            server=Config.MT5_SERVER,
            path=Config.MT5_PATH,
        )

        tunnel_strategy.run_strategy(
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

def setup_schedule():
    schedule.every().day.at("09:00").do(initialize_strategy)
    logging.info("Scheduler setup complete. Next run at: %s", schedule.next_run())
