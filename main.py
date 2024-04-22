import time
import schedule
from config import Config
from metatrader import mt5_lib
from strategy import tunnel_strategy

if __name__ == "__main__":
    print("Running on: ", Config.MT5_SERVER, "Metatrader 5")

    mt5_init  = mt5_lib.Mt5(
        login=Config.MT5_LOGIN, 
        password=Config.MT5_PASSWORD, 
        server=Config.MT5_SERVER, 
        path=Config.MT5_PATH,
    )

    tunnel_strategy.run_strategy(
        symbols =Config.SYMBOLS,  
        mt5_init = mt5_init, 
        timeframe = Config.MT5_TIMEFRAME, 
        lot_size= Config.MT5_LOT_SIZE, 
        min_take_profit =  Config.MIN_TP_PROFIT, 
        max_loss_per_day= Config.MAX_LOSS_PER_DAY, 
        starting_equity = Config.STARTING_EQUITY,  
        max_traders_per_day= Config.LIMIT_NO_OF_TRADES
    )
    

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            print(f"Exception: {e}")
        time.sleep(1)
