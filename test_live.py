import MetaTrader5 as mt5
import pandas as pd
import logging
from datetime import datetime, timedelta
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute data
EMA_PERIOD = 10
TRADE_VOLUME = 0.1  # Lot size

# Initialize MetaTrader 5 connection
def connect_mt5():
    if not mt5.initialize():
        logging.error("Failed to initialize MetaTrader5")
        return False
    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to retrieve account info")
        return False
    logging.info(f"Connected to account {account_info.login}")
    return True

# Get historical data (for EMA calculation)
def get_price_data(symbol, timeframe, num_bars):
    utc_from = datetime.now() - timedelta(minutes=num_bars)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, num_bars)
    if rates is None or len(rates) == 0:
        logging.error(f"Failed to retrieve data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Calculate EMA
def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()

# Enter a trade
def enter_trade(action, symbol, volume):
    price = mt5.symbol_info_tick(symbol).ask if action == 'buy' else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "EMA Strategy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Failed to place {action} order: {result.retcode}")
    else:
        logging.info(f"Placed {action} order at price {price} for {volume} lots.")

# Main trading loop
def main():
    if not connect_mt5():
        return

    while True:
        # Get price data for the last 20 minutes (for EMA calculation)
        data = get_price_data(SYMBOL, TIMEFRAME, EMA_PERIOD + 5)
        if data is None:
            time.sleep(60)
            continue

        # Calculate the EMA
        data['EMA'] = calculate_ema(data, EMA_PERIOD)
        current_price = data['close'].iloc[-1]
        current_ema = data['EMA'].iloc[-1]
        
        logging.info(f"Current price: {current_price}, EMA: {current_ema}")

        # Buy if the price crosses above the EMA
        if current_price > current_ema:
            logging.info("Price is above EMA, placing buy order.")
            enter_trade('buy', SYMBOL, TRADE_VOLUME)
        # Sell if the price crosses below the EMA
        elif current_price < current_ema:
            logging.info("Price is below EMA, placing sell order.")
            enter_trade('sell', SYMBOL, TRADE_VOLUME)

        # Sleep for a minute before checking again (since we use 1-minute candles)
        time.sleep(60)

    # Shut down the MT5 connection
    mt5.shutdown()

if __name__ == "__main__":
    main()
