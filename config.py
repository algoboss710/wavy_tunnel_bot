import os
from dotenv import load_dotenv
from utils.error_handling import handle_error, critical_error
import logging

# Load environment variables from the .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
dotenv_path = os.path.join(project_dir, '.env')

def reload_env():
    load_dotenv(dotenv_path, override=True)

reload_env()

class Config:
    MT5_LOGIN = os.getenv("MT5_LOGIN")
    if not MT5_LOGIN:
        raise ValueError("MT5_LOGIN environment variable is not set.")

    MT5_PASSWORD = os.getenv("MT5_PASSWORD")
    if not MT5_PASSWORD:
        raise ValueError("MT5_PASSWORD environment variable is not set.")

    MT5_SERVER = os.getenv("MT5_SERVER")
    if not MT5_SERVER:
        raise ValueError("MT5_SERVER environment variable is not set.")

    MT5_PATH = os.getenv("MT5_PATH")
    if not MT5_PATH:
        raise ValueError("MT5_PATH environment variable is not set.")

    MT5_TIMEFRAME = os.getenv("MT5_TIMEFRAME")
    if MT5_TIMEFRAME not in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
        raise ValueError(f"Invalid MT5_TIMEFRAME value: {MT5_TIMEFRAME}. Expected values: M1, M5, M15, M30, H1, H4, D1.")

    SYMBOLS = os.getenv("SYMBOLS")
    if SYMBOLS:
        SYMBOLS = SYMBOLS.split(",")
    else:
        raise ValueError("SYMBOLS environment variable is not set.")

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_IDS = os.getenv("TELEGRAM_IDS")
    if TELEGRAM_TOKEN and TELEGRAM_IDS:
        TELEGRAM_IDS = TELEGRAM_IDS.split(",")
    else:
        TELEGRAM_TOKEN = None
        TELEGRAM_IDS = None

    try:
        MIN_TP_PROFIT = float(os.getenv("MIN_TP_PROFIT", 50.0))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid MIN_TP_PROFIT value: {os.getenv('MIN_TP_PROFIT')}. Expected a numeric value.")

    try:
        MAX_LOSS_PER_DAY = float(os.getenv("MAX_LOSS_PER_DAY", 1000.0))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid MAX_LOSS_PER_DAY value: {os.getenv('MAX_LOSS_PER_DAY')}. Expected a numeric value.")

    try:
        STARTING_EQUITY = float(os.getenv("STARTING_EQUITY", 10000.0))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid STARTING_EQUITY value: {os.getenv('STARTING_EQUITY')}. Expected a numeric value.")

    try:
        LIMIT_NO_OF_TRADES = int(os.getenv("LIMIT_NO_OF_TRADES", 5))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid LIMIT_NO_OF_TRADES value: {os.getenv('LIMIT_NO_OF_TRADES')}. Expected an integer value.")

    try:
        RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
    except ValueError:
        raise ValueError(f"Invalid RISK_PER_TRADE value: {os.getenv('RISK_PER_TRADE')}. Expected a numeric value.")

    if not 0 < RISK_PER_TRADE <= 1:
        raise ValueError(f"RISK_PER_TRADE value must be between 0 and 1. Current value: {RISK_PER_TRADE}")

    try:
        PIP_VALUE = float(os.getenv("PIP_VALUE", 1))
    except ValueError:
        raise ValueError(f"Invalid PIP_VALUE value: {os.getenv('PIP_VALUE')}. Expected a numeric value.")

    try:
        MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", 0.2))
    except ValueError:
        raise ValueError(f"Invalid MAX_DRAWDOWN value: {os.getenv('MAX_DRAWDOWN')}. Expected a numeric value.")

    # Configuration for enabling pending order fallback
    ENABLE_PENDING_ORDER_FALLBACK = os.getenv("ENABLE_PENDING_ORDER_FALLBACK", "True").lower() in ("true", "1", "yes")

    # Configuration for SL/TP adjustment pips
    SL_TP_ADJUSTMENT_PIPS = float(os.getenv("SL_TP_ADJUSTMENT_PIPS", 0.0001))

    @classmethod
    def validate(cls):
        try:
            # Ensure all required environment variables are set
            required_vars = [
                'MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER', 'MT5_PATH',
                'MT5_TIMEFRAME', 'SYMBOLS'
            ]
            for var in required_vars:
                if not getattr(cls, var, None):
                    raise ValueError(f"Missing required environment variable: {var}")

            # Ensure all numeric variables have valid numeric values
            numeric_vars = ['MIN_TP_PROFIT', 'MAX_LOSS_PER_DAY', 'STARTING_EQUITY', 'RISK_PER_TRADE', 'PIP_VALUE', 'MAX_DRAWDOWN', 'SL_TP_ADJUSTMENT_PIPS']
            for var in numeric_vars:
                if not isinstance(getattr(cls, var, None), (int, float)):
                    raise ValueError(f"Invalid value for {var}. Expected a numeric value.")

            # Validate integer-specific configuration
            if not isinstance(cls.LIMIT_NO_OF_TRADES, int):
                raise ValueError(f"Invalid value for LIMIT_NO_OF_TRADES. Expected an integer value.")

        except ValueError as e:
            handle_error(e, "Configuration validation failed")
            critical_error(e, "Invalid configuration settings")

    @classmethod
    def log_config(cls):
        # Log all configuration settings for debugging purposes
        for attr, value in cls.__dict__.items():
            if not callable(value) and not attr.startswith("__") and not isinstance(value, classmethod):
                logging.info(f"{attr}: {value}")

# Validate configuration and log it
try:
    Config.validate()
    Config.log_config()
except Exception as e:
    handle_error(e, "Error occurred during configuration validation")
    raise
