from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize():
    print("Failed to initialize MT5")
    exit()

symbols = ["XAUUSD", "EURUSD"]
timeframes = {"M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

for symbol in symbols:
    for tf_name, tf in timeframes.items():
        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        if rates is None or len(rates) == 0:
            print(f"No data for {symbol} {tf_name}")
        else:
            print(f"Data available for {symbol} {tf_name}: {len(rates)} bars")

mt5.shutdown()
