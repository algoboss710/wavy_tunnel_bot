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
