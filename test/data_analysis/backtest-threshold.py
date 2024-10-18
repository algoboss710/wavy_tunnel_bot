import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import json
import concurrent.futures

# Initialize MetaTrader5 connection
if not mt5.initialize():
    print("MetaTrader5 initialization failed")
    mt5.shutdown()
    quit()

# Provided ATR values converted into threshold ranges
threshold_ranges = {
    "EURUSD": {
        "M15": [0.00017805096278591235, 0.0003561019255718247, 0.000534152888357737, 0.0007122038511436494, 0.0008902548139295617, 0.001068305776715474],
        "H1": [0.0003664142080071423, 0.0007328284160142846, 0.0010992426240214268, 0.0014656568320285692, 0.0018320710400357116, 0.002198485248042854],
        "H4": [0.0007649220508407253, 0.0015298441016814506, 0.002294766152522176, 0.0030596882033629013, 0.0038246102542036267, 0.004589532305044352],
        "D1": [0.0020335066305003025, 0.004067013261000605, 0.0061005198915009075, 0.00813402652200121, 0.010167533152501513, 0.012201039783001815]
    },
    "GBPUSD": {
        "M15": [0.0002377139737810847, 0.0004754279475621694, 0.000713141921343254, 0.0009508558951243388, 0.0011885698689054235, 0.001426283842686508],
        "H1": [0.00048697861730619765, 0.0009739572346123953, 0.001460935851918593, 0.0019479144692247907, 0.0024348930865309883, 0.002921871703837186],
        "H4": [0.0010110135873162976, 0.0020220271746325953, 0.0030330407619488926, 0.004044054349265191, 0.005055067936581488, 0.006066081523897785],
        "D1": [0.0026617440225035157, 0.0053234880450070314, 0.007985232067510547, 0.010646976090014063, 0.013308720112517578, 0.015970464135021094]
    },
    "USDJPY": {
        "M15": [0.028751486631933025, 0.05750297326386605, 0.08625445989579908, 0.1150059465277321, 0.14375743315966513, 0.17250891979159816],
        "H1": [0.05980898581513389, 0.11961797163026779, 0.17942695744540167, 0.23923594326053558, 0.29904492907566946, 0.35885391489080335],
        "H4": [0.12786571974712032, 0.25573143949424065, 0.38359715924136097, 0.5114628789884813, 0.6393285987356016, 0.7671943184827219],
        "D1": [0.34025015069318864, 0.6805003013863773, 1.020750452079566, 1.361000602772755, 1.7012507534659434, 2.041500904159132]
    },
    "AUDUSD": {
        "M15": [0.0001655621054240763, 0.0003311242108481526, 0.000496686316272229, 0.0006622484216963053, 0.0008278105271203816, 0.000993372632544458],
        "H1": [0.00033993725924578855, 0.0006798745184915771, 0.0010198117777373656, 0.0013597490369831542, 0.0016996862962289427, 0.0020396235554747312],
        "H4": [0.0007129633092810808, 0.0014259266185621617, 0.0021388899278432425, 0.0028518532371243234, 0.0035648165464054042, 0.004277779855686485],
        "D1": [0.0018219411794253568, 0.0036438823588507137, 0.005465823538276071, 0.007287764717701428, 0.009109705897126784, 0.010931647076552141]
    }
}
def get_historical_data(symbol, timeframe, start_date, end_date):
    timezone = pytz.timezone("Etc/UTC")
    start_date = timezone.localize(start_date)
    end_date = timezone.localize(end_date)

    # Fetch historical data from MetaTrader 5
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"Failed to retrieve data for {symbol} on {timeframe_to_string(timeframe)} timeframe")
        return None

    # Create DataFrame from the rates
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Log the historical data retrieved for inspection
    print(f"Retrieved {len(df)} data points for {symbol} on {timeframe_to_string(timeframe)} timeframe.")

    return df


# Modify the function to accept a threshold dynamically
def generate_signals(df, symbol, threshold):
    threshold_in_price_units = threshold * mt5.symbol_info(symbol).point  # Convert threshold to price units

    df['max_wavy'] = df[['ema_34', 'close']].max(axis=1)
    df['min_wavy'] = df[['ema_34', 'close']].min(axis=1)
    df['max_tunnel'] = df[['ema_144', 'ema_169']].max(axis=1)
    df['min_tunnel'] = df[['ema_144', 'ema_169']].min(axis=1)

    # Long entry condition
    df['long_entry'] = (df['open'] > df['max_wavy']) & (df['min_wavy'] > df['max_tunnel']) & (df['rsi'] < 70)
    df['long_entry'] &= (df['open'] > df['max_wavy'] + threshold_in_price_units)

    # Short entry condition
    df['short_entry'] = (df['open'] < df['min_wavy']) & (df['max_wavy'] < df['min_tunnel']) & (df['rsi'] > 30)
    df['short_entry'] &= (df['open'] < df['min_wavy'] - threshold_in_price_units)

    # Exit conditions
    df['long_exit'] = df['close'] < df['min_wavy']
    df['short_exit'] = df['close'] > df['max_wavy']

    return df

# Backtest function with dynamic threshold for each currency pair and timeframe
def run_backtest(df, symbol, timeframe, threshold, initial_balance=10000, risk_per_trade=0.01):
    balance = initial_balance
    position = 0
    trades = []

    # Add logging for entry/exit signals
    print(f"Running backtest for {symbol} on {timeframe_to_string(timeframe)} with threshold {threshold}.")

    # Generate signals
    df = generate_signals(df, symbol, threshold)

    for i in range(1, len(df)):
        if position == 0:
            if df['long_entry'].iloc[i]:
                position = 1  # Long position
                entry_price = df['open'].iloc[i]
                print(f"Opening long position for {symbol} at {entry_price} on {df.index[i]}")
            elif df['short_entry'].iloc[i]:
                position = -1  # Short position
                entry_price = df['open'].iloc[i]
                print(f"Opening short position for {symbol} at {entry_price} on {df.index[i]}")

        # Exit conditions
        if position == 1 and df['long_exit'].iloc[i]:
            exit_price = df['close'].iloc[i]
            profit = (exit_price - entry_price) * risk_per_trade
            balance += profit
            trades.append({'entry': entry_price, 'exit': exit_price, 'profit': profit})
            print(f"Closing long position for {symbol} at {exit_price} on {df.index[i]}")
            position = 0

        if position == -1 and df['short_exit'].iloc[i]:
            exit_price = df['close'].iloc[i]
            profit = (entry_price - exit_price) * risk_per_trade
            balance += profit
            trades.append({'entry': entry_price, 'exit': exit_price, 'profit': profit})
            print(f"Closing short position for {symbol} at {exit_price} on {df.index[i]}")
            position = 0

    return balance, trades

# Helper to convert timeframe to string
def timeframe_to_string(timeframe):
    timeframe_dict = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1"
    }
    return timeframe_dict.get(timeframe, str(timeframe))

# Run multiple backtests concurrently using threads
def run_threshold_tests_concurrently(symbol, timeframes, start_date, end_date, threshold_ranges):
    results = {}

    def run_test(symbol, timeframe, threshold):
        # Fetch historical data
        df = get_historical_data(symbol, timeframe, start_date, end_date)
        if df is not None:
            balance, trades = run_backtest(df, symbol, timeframe, threshold)
            return (symbol, timeframe, threshold, balance, trades)
        return None

    # Use ThreadPoolExecutor to run multiple tests concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for timeframe in timeframes:
            tf_string = timeframe_to_string(timeframe)
            for threshold in threshold_ranges[symbol][tf_string]:
                futures.append(executor.submit(run_test, symbol, timeframe, threshold))

        # Collect results as they are completed
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                symbol, timeframe, threshold, balance, trades = result
                tf_string = timeframe_to_string(timeframe)
                results.setdefault(symbol, {}).setdefault(tf_string, []).append({
                    'threshold': threshold,
                    'balance': balance,
                    'trades': len(trades)
                })

    return results

# Function to log the results
def log_results(results):
    with open('threshold_test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to 'threshold_test_results.json'")

# Define symbols, timeframes, and date range
symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
timeframes = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 10, 15)

# Run concurrent tests for all symbols
all_results = {}
for symbol in symbols:
    print(f"\nStarting threshold tests for {symbol}")
    all_results[symbol] = run_threshold_tests_concurrently(symbol, timeframes, start_date, end_date, threshold_ranges)

# Log the final results
log_results(all_results)

# Shutdown MetaTrader 5 connection
mt5.shutdown()
