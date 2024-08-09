import MetaTrader5 as mt5
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to MetaTrader 5
def connect_mt5():
    if not mt5.initialize():
        logging.error("Failed to initialize MetaTrader5")
        return False
    logging.info("MetaTrader5 initialized successfully.")
    return True

# Get latest tick data
def get_latest_tick(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        logging.info(f"Latest tick data for {symbol}: Time: {datetime.fromtimestamp(tick.time)}, Bid: {tick.bid}, Ask: {tick.ask}")
        return tick
    else:
        logging.error(f"Failed to retrieve tick data for {symbol}")
        return None

# Place a market order
def place_market_order(symbol, action, volume):
    tick = get_latest_tick(symbol)
    if not tick:
        logging.error("No tick data available. Cannot place order.")
        return None

    order_type = mt5.ORDER_TYPE_BUY if action.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
    price = tick.ask if action.upper() == 'BUY' else tick.bid

    order = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': volume,
        'type': order_type,
        'price': price,
        'deviation': 10,
        'magic': 12345,
        'comment': "Simple Market Order",
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_FOK,
    }

    logging.info(f"Placing {action} order for {symbol} at price {price}")
    result = mt5.order_send(order)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"Trade executed successfully: {result}")
    else:
        logging.error(f"Trade failed with retcode {result.retcode}: {result.comment}")

    return result

# Main function
def main():
    symbol = "EURUSD"
    action = "BUY"
    volume = 0.01

    # Connect to MT5
    if not connect_mt5():
        return

    # Ensure the symbol is visible and active
    if not mt5.symbol_select(symbol, True):
        logging.error(f"Failed to select symbol {symbol}.")
        mt5.shutdown()
        return

    # Place a market order
    result = place_market_order(symbol, action, volume)

    if result:
        logging.info(f"Order result: {result}")

    # Shut down the MT5 connection
    mt5.shutdown()
    logging.info("MetaTrader5 connection shut down.")

if __name__ == "__main__":
    main()
