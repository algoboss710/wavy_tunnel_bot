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

    # Retrieve historical data for the symbol
    data = get_historical_data(symbol, timeframe, start_time, end_time)
    if data is not None:
        logging.info(f"Data for {symbol} retrieved and cleaned successfully.")

        # Save data to CSV in the 'csv' directory
        csv_path = os.path.join(csv_dir, f"{symbol}_historical_data.csv")
        data.to_csv(csv_path, index=False)
        logging.info(f"Data for {symbol} saved to {csv_path}.")

    # Shutdown MT5 connection
    shutdown_mt5()

if __name__ == '__main__':
    main()
