import MetaTrader5 as mt5
import logging
from datetime import datetime, time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to MetaTrader 5
def connect_mt5():
    if not mt5.initialize():
        logging.error("Failed to initialize MetaTrader5")
        return False
    logging.info("MetaTrader5 initialized successfully.")
    return True

# Get symbol market hours
def get_symbol_market_hours(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to retrieve symbol info for {symbol}")
        return None

    logging.info(f"Market hours for {symbol}:")

    for i in range(7):  # For each day of the week (0 = Monday, 6 = Sunday)
        session_open = symbol_info.session_deals[i].open if symbol_info.session_deals else None
        session_close = symbol_info.session_deals[i].close if symbol_info.session_deals else None

        if session_open and session_close:
            open_time = time(session_open // 3600, (session_open % 3600) // 60)
            close_time = time(session_close // 3600, (session_close % 3600) // 60)
            logging.info(f"Day {i}: Open: {open_time}, Close: {close_time}")
        else:
            logging.info(f"Day {i}: Market closed")

# Main function
def main():
    symbol = "EURUSD"

    # Connect to MT5
    if not connect_mt5():
        return

    # Retrieve and display market hours
    get_symbol_market_hours(symbol)

    # Shut down the MT5 connection
    mt5.shutdown()
    logging.info("MetaTrader5 connection shut down.")

if __name__ == "__main__":
    main()
