import logging

def setup_logging():
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('error.log')
        ]
    )

setup_logging()

def handle_error(error, message):
    logging.error(f"{message}: {str(error)}")

def critical_error(error, message):
    logging.critical(f"{message}: {str(error)}")
    raise SystemExit(1)

def warn_error(error, message):
    logging.warning(f"{message}: {str(error)}")