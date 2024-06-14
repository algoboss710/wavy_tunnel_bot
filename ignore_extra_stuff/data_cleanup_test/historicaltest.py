import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os
import logging
from utils.error_handling import handle_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_mt5():
    if not mt5.initialize():
        logging.error("Failed to initialize MetaTrader5")
        return False
    logging.info("MetaTrader 5 initialized successfully.")
    return True

def shutdown_mt5():
    mt5.shutdown()
    logging.info("MetaTrader 5 connection shut down.")

def get_historical_data(symbol, timeframe, start_time, end_time):
    try:
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "H1": mt5.TIMEFRAME_H1,
            "D1": mt5.TIMEFRAME_D1
        }
        timeframe_mt5 = timeframe_map.get(timeframe)
        
        if not timeframe_mt5:
            logging.error(f"Invalid time frame: {timeframe}")
            return None

        rates = mt5.copy_rates_range(symbol, timeframe_mt5, start_time, end_time)
        if rates is None or len(rates) == 0:
            raise ValueError(f"Failed to retrieve historical data for {symbol} with timeframe {timeframe} from {start_time} to {end_time}")
        
        data = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
        data['time'] = pd.to_datetime(data['time'], unit='s')
        
        logging.info(f"Historical data shape after retrieval: {data.shape}")
        logging.info(f"Historical data head after retrieval:\n{data.head()}")
        
        if data.empty:
            raise ValueError(f"No historical data retrieved for {symbol} with timeframe {timeframe} from {start_time} to {end_time}")

        # Data Cleaning
        data = data.dropna()  # Drop missing values
        data = data[(data['open'] > 0) & (data['high'] > 0) & (data['low'] > 0) & (data['close'] > 0)]  # Remove zero/negative prices
        
        return data
    except Exception as e:
        handle_error(e, f"Failed to retrieve historical data for {symbol}")
        return None

def calculate_indicators(data):
    data['wavy_h'] = data['high'].ewm(span=34).mean()
    data['wavy_c'] = data['close'].ewm(span=34).mean()
    data['wavy_l'] = data['low'].ewm(span=34).mean()
    data['tunnel1'] = data['close'].ewm(span=144).mean()
    data['tunnel2'] = data['close'].ewm(span=169).mean()
    data['long_term_ema'] = data['close'].ewm(span=200).mean()
    return data

def detect_peaks_and_dips(data, peak_type):
    data['peak'] = False
    data['dip'] = False
    
    for i in range(peak_type, len(data) - peak_type):
        center_index = i
        is_peak = True
        is_dip = True
        
        for j in range(1, peak_type + 1):
            if data['high'][center_index] <= data['high'][center_index - j] or data['high'][center_index] <= data['high'][center_index + j]:
                is_peak = False
            if data['low'][center_index] >= data['low'][center_index - j] or data['low'][center_index] >= data['low'][center_index + j]:
                is_dip = False
        
        if is_peak:
            data.at[center_index, 'peak'] = True
        if is_dip:
            data.at[center_index, 'dip'] = True

    return data

def check_entry_conditions(row, peaks, dips, symbol):
    buy_condition = (
        row['open'] > max(row['wavy_c'], row['wavy_h'], row['wavy_l']) and
        min(row['wavy_c'], row['wavy_h'], row['wavy_l']) > max(row['tunnel1'], row['tunnel2'])
    )

    sell_condition = (
        row['open'] < min(row['wavy_c'], row['wavy_h'], row['wavy_l']) and
        max(row['wavy_c'], row['wavy_h'], row['wavy_l']) < min(row['tunnel1'], row['tunnel2'])
    )

    return buy_condition, sell_condition

def check_exit_conditions(row, position, symbol):
    exit_condition = False
    if position['type'] == 'BUY':
        exit_condition = (
            row['close'] < min(row['wavy_c'], row['wavy_h'], row['wavy_l']) or
            row['close'] > max(row['tunnel1'], row['tunnel2'])
        )
    elif position['type'] == 'SELL':
        exit_condition = (
            row['close'] > max(row['wavy_c'], row['wavy_h'], row['wavy_l']) or
            row['close'] < min(row['tunnel1'], row['tunnel2'])
        )
    return exit_condition

def calculate_total_profit(trades):
    total_profit = sum(trade['exit_price'] - trade['entry_price'] for trade in trades if 'exit_price' in trade)
    return total_profit

def calculate_win_rate(trades):
    wins = sum(1 for trade in trades if 'exit_price' in trade and trade['exit_price'] > trade['entry_price'])
    win_rate = wins / len(trades) if trades else 0
    return win_rate

def calculate_max_drawdown(trades, initial_balance):
    peak_balance = initial_balance
    drawdown = 0
    max_drawdown = 0
    balance = initial_balance

    for trade in trades:
        if 'exit_price' in trade:
            profit = trade['exit_price'] - trade['entry_price']
            balance += profit
            if balance > peak_balance:
                peak_balance = balance
            drawdown = peak_balance - balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return max_drawdown

def calculate_metrics(trades, initial_balance):
    total_profit = calculate_total_profit(trades)
    win_rate = calculate_win_rate(trades)
    max_drawdown = calculate_max_drawdown(trades, initial_balance)
    
    metrics = {
        "Total Profit": total_profit,
        "Win Rate": win_rate,
        "Maximum Drawdown": max_drawdown
    }
    
    return metrics

def main():
    # Initialize MT5
    if not initialize_mt5():
        return

    # Create a directory for CSV files
    csv_dir = "csv"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    symbol = "EURUSD"
    timeframe = "H1"
    start_time = datetime(2023, 1, 1)
    end_time = datetime.now()
    initial_balance = 10000  # Example initial balance

    # Retrieve historical data for the symbol
    data = get_historical_data(symbol, timeframe, start_time, end_time)
    if data is not None:
        logging.info(f"Data for {symbol} retrieved and cleaned successfully.")

        # Calculate indicators
        data = calculate_indicators(data)

        # Detect peaks and dips
        data = detect_peaks_and_dips(data, peak_type=21)

        # Simulate trading
        trades = []
        position = None

        for i in range(len(data)):
            row = data.iloc[i]
            buy_condition, sell_condition = check_entry_conditions(row, data['peak'], data['dip'], symbol)

            if buy_condition and position is None:
                position = {
                    'type': 'BUY',
                    'entry_price': row['close'],
                    'entry_time': row['time']
                }
                logging.info(f"Buy condition met at {row['time']} with price {row['close']}")

            elif sell_condition and position is None:
                position = {
                    'type': 'SELL',
                    'entry_price': row['close'],
                    'entry_time': row['time']
                }
                logging.info(f"Sell condition met at {row['time']} with price {row['close']}")

            if position is not None and check_exit_conditions(row, position, symbol):
                position['exit_price'] = row['close']
                position['exit_time'] = row['time']
                trades.append(position)
                logging.info(f"Trade closed at {row['time']} with price {row['close']}")
                position = None

        # Calculate performance metrics
        metrics = calculate_metrics(trades, initial_balance)
        logging.info(f"Performance metrics: {metrics}")

        # Save data to CSV in the 'csv' directory
        csv_path = os.path.join(csv_dir, f"{symbol}_historical_data_with_indicators_and_peaks_dips.csv")
        data.to_csv(csv_path, index=False)
        logging.info(f"Data for {symbol} saved to {csv_path}.")

    # Shutdown MT5 connection
    shutdown_mt5()

if __name__ == '__main__':
    main()
