import MetaTrader5 as mt5
import pandas as pd
import requests_cache
from datetime import datetime, timedelta
from requests_cache import CachedSession
from utils.error_handling import handle_error


def initialize_mt5():
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        return False
    return True

def shutdown_mt5():
    mt5.shutdown()

start_time = datetime.now() - timedelta(days=30)  # Example: 30 days ago
end_time = datetime.now()  # Current time

def get_historical_data(symbol, timeframe, start_time, end_time):
    try:
        # Retrieve data from MT5
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        if rates is None or len(rates) == 0:
            raise ValueError(f"Failed to retrieve historical data for {symbol} with timeframe {timeframe} from {start_time} to {end_time}")
        
        data = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
        data['time'] = pd.to_datetime(data['time'], unit='s')
        
        print(f"Historical data shape after retrieval: {data.shape}")
        print(f"Historical data head after retrieval:\n{data.head()}")
        
        if data.empty:
            raise ValueError(f"No historical data retrieved for {symbol} with timeframe {timeframe} from {start_time} to {end_time}")

        # Data Cleaning
        data = data.dropna()  # Drop missing values
        data = data[(data['open'] > 0) & (data['high'] > 0) & (data['low'] > 0) & (data['close'] > 0)]  # Remove zero/negative prices
        
        return data
    except Exception as e:
        handle_error(e, f"Failed to retrieve historical data for {symbol}")
        return None

def retrieve_historical_data(symbol, start_date, end_date, timeframe):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    data = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

def get_current_price(symbol):
    prices = mt5.symbol_info_tick(symbol)
    if prices:
        return prices.last
    else:
        print(f"Failed to retrieve current price for {symbol}")
        return None

def get_account_info():
    account_info = mt5.account_info()
    if account_info:
        return account_info._asdict()
    else:
        print("Failed to retrieve account information")
        return None

def get_available_symbols():
    symbols = mt5.symbols_get()
    if symbols:
        return [symbol.name for symbol in symbols]
    else:
        print("Failed to retrieve available symbols")
        return None

def get_symbol_info(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        return symbol_info._asdict()
    else:
        print(f"Failed to retrieve information for {symbol}")
        return None

def get_positions():
    positions = mt5.positions_get()
    if positions:
        positions_data = []
        for position in positions:
            position_data = position._asdict()
            positions_data.append(position_data)
        return positions_data
    else:
        print("No open positions found")
        return None

def get_orders():
    orders = mt5.orders_get()
    if orders:
        orders_data = []
        for order in orders:
            order_data = order._asdict()
            orders_data.append(order_data)
        return orders_data
    else:
        print("No pending orders found")
        return None

if __name__ == '__main__':
    if initialize_mt5():
        symbol = "EURUSD"
        timeframe = mt5.TIMEFRAME_H1
        start_time = datetime(2023, 1, 1)
        end_time = datetime.now()

        historical_data = get_historical_data(symbol, timeframe, start_time, end_time)
        if historical_data is not None:
            print(f"Historical data for {symbol}:")
            print(historical_data.head())

        current_price = get_current_price(symbol)
        if current_price is not None:
            print(f"Current price for {symbol}: {current_price}")

        account_info = get_account_info()
        if account_info is not None:
            print("Account information:")
            print(account_info)

        available_symbols = get_available_symbols()
        if available_symbols is not None:
            print("Available symbols:")
            print(available_symbols)

        symbol_info = get_symbol_info(symbol)
        if symbol_info is not None:
            print(f"Symbol information for {symbol}:")
            print(symbol_info)

        positions = get_positions()
        if positions is not None:
            print("Open positions:")
            print(positions)

        orders = get_orders()
        if orders is not None:
            print("Pending orders:")
            print(orders)

        shutdown_mt5()