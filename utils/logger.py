import logging
import os

def setup_logging(log_level=logging.INFO, log_file="app.log"):
    """
    Set up logging configuration.
    """
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    # Ensure no duplicate handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])

    logging.info("Logging setup complete.")

def log_bid_ask_prices(symbol, bid, ask):
    """
    Log the bid and ask prices for a given symbol.
    """
    logging.info(f"Symbol: {symbol} | Bid: {bid} | Ask: {ask}")

def log_trade_attempt(symbol, attempt_number, max_attempts):
    """
    Log an attempt to place a trade.
    """
    logging.info(f"Attempting to place trade for {symbol} (Attempt {attempt_number}/{max_attempts})")

def log_symbol_subscription(symbol):
    """
    Log the subscription status of a symbol.
    """
    logging.info(f"Symbol {symbol} is already subscribed and visible.")
