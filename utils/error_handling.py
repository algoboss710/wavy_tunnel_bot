import logging

# Setup the logging configuration in this module if not already configured
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_error(err, message="An error occurred"):
    """ General error handling function. """
    logging.error(f"{message}: {str(err)}")

def critical_error(err, message="A critical error occurred", exit_code=1):
    """ Handle critical errors that require shutting down the application. """
    logging.critical(f"{message}: {str(err)}")
    exit(exit_code)

def warn_error(err, message="A warning error occurred"):
    """ Log warnings that do not require stopping the application. """
    logging.warning(f"{message}: {str(err)}")
