import os
from dotenv import load_dotenv
from utils.error_handling import handle_error, critical_error
import logging
import MetaTrader5 as mt5
from datetime import datetime

# Load environment variables from the .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
dotenv_path = os.path.join(project_dir, '.env')

def reload_env():
    load_dotenv(dotenv_path, override=True)

reload_env()

class Config:
    # MetaTrader 5 Settings
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
    TIMEFRAME_DICT = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    if MT5_TIMEFRAME not in TIMEFRAME_DICT:
        raise ValueError(f"Invalid MT5_TIMEFRAME value: {MT5_TIMEFRAME}. Expected values: M1, M5, M15, M30, H1, H4, D1.")
    MT5_TIMEFRAME_VALUE = TIMEFRAME_DICT[MT5_TIMEFRAME]

    SYMBOLS = os.getenv("SYMBOLS")
    if SYMBOLS:
        SYMBOLS = SYMBOLS.split(",")
    else:
        raise ValueError("SYMBOLS environment variable is not set.")

    # Data Retrieval Settings
    DATA_SOURCE = os.getenv("DATA_SOURCE", "MT5")
    HISTORICAL_DATA_CANDLES = int(os.getenv("HISTORICAL_DATA_CANDLES", 200))

    # Telegram Bot Settings
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_IDS = os.getenv("TELEGRAM_IDS")
    if TELEGRAM_TOKEN and TELEGRAM_IDS:
        TELEGRAM_IDS = TELEGRAM_IDS.split(",")
    else:
        TELEGRAM_TOKEN = None
        TELEGRAM_IDS = None

    # Trading Strategy Settings
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

    ENABLE_PENDING_ORDER_FALLBACK = os.getenv("ENABLE_PENDING_ORDER_FALLBACK", "True").lower() in ("true", "1", "yes")

    # SL/TP Adjustment
    SL_TP_ADJUSTMENT_PIPS = float(os.getenv("SL_TP_ADJUSTMENT_PIPS", 0.0001))

    # New Strategy Parameters
    PEAK_DETECTION_WINDOW = int(os.getenv("PEAK_DETECTION_WINDOW", 21))

    # Backtesting Settings
    BACKTEST_SLIPPAGE = float(os.getenv("BACKTEST_SLIPPAGE", 0.0))
    BACKTEST_TRANSACTION_COST = float(os.getenv("BACKTEST_TRANSACTION_COST", 0.0))

    # Optional Backtest Start/End Dates for Backtesting
    BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE")
    BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE")

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
            numeric_vars = ['MIN_TP_PROFIT', 'MAX_LOSS_PER_DAY', 'STARTING_EQUITY', 'RISK_PER_TRADE', 'PIP_VALUE', 'MAX_DRAWDOWN', 'SL_TP_ADJUSTMENT_PIPS', 'BACKTEST_SLIPPAGE', 'BACKTEST_TRANSACTION_COST']
            for var in numeric_vars:
                if not isinstance(getattr(cls, var, None), (int, float)):
                    raise ValueError(f"Invalid value for {var}. Expected a numeric value.")

            # Validate integer-specific configuration
            integer_vars = ['LIMIT_NO_OF_TRADES', 'HISTORICAL_DATA_CANDLES', 'PEAK_DETECTION_WINDOW']
            for var in integer_vars:
                if not isinstance(getattr(cls, var), int) or getattr(cls, var) <= 0:
                    raise ValueError(f"Invalid value for {var}. Expected a positive integer.")

            # Validate backtesting date configuration
            if cls.BACKTEST_START_DATE and cls.BACKTEST_END_DATE:
                try:
                    start_date = datetime.strptime(cls.BACKTEST_START_DATE, "%Y-%m-%d")
                    end_date = datetime.strptime(cls.BACKTEST_END_DATE, "%Y-%m-%d")
                    if start_date > end_date:
                        raise ValueError("BACKTEST_START_DATE must be earlier than BACKTEST_END_DATE.")
                    if end_date > datetime.now():
                        raise ValueError("BACKTEST_END_DATE cannot be in the future.")
                except ValueError as e:
                    raise ValueError(f"Invalid backtest date format: {str(e)}. Use YYYY-MM-DD format.")
            elif cls.BACKTEST_START_DATE or cls.BACKTEST_END_DATE:
                raise ValueError("Both BACKTEST_START_DATE and BACKTEST_END_DATE must be set for backtesting.")

            # Validate data source
            if cls.DATA_SOURCE not in ["MT5", "CSV", "API"]:
                raise ValueError(f"Invalid DATA_SOURCE value: {cls.DATA_SOURCE}. Expected 'MT5', 'CSV', or 'API'.")

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