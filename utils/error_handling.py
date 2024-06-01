import logging

def handle_error(error, message):
    logging.error(f"{message}: {str(error)}")

def critical_error(error, message):
    logging.critical(f"{message}: {str(error)}")
    raise SystemExit(1)

def warn_error(error, message):
    logging.warning(f"{message}: {str(error)}")
