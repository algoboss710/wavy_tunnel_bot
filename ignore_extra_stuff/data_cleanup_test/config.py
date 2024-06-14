# config.py
class Config:
    MT5_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"  # Update with the correct path to your MT5 terminal
    SYMBOLS = ["EURUSD", "GBPUSD"]  # List of instruments
    TIMEFRAME = "H1"  # Time frame, e.g., "H1", "D1"
    START_DATE = "2023-01-01"
    END_DATE = "2023-01-30"
    PIP_VALUE = 0.0001  # Example pip value
    STARTING_EQUITY = 10000
    RISK_PER_TRADE = 0.01
    MIN_TP_PROFIT = 0.0020
    MAX_LOSS_PER_DAY = 0.02
    LIMIT_NO_OF_TRADES = 10
