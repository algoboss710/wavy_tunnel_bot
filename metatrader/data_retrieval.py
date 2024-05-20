import MetaTrader5 as mt5
import pandas as pd
import requests_cache
from datetime import datetime
from utils.error_handling import handle_error


def initialize_mt5():
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        return False
    return True

def shutdown_mt5():
    mt5.shutdown()

requests_cache.install_cache('historical_data_cache', backend='sqlite', expire_after=3600)


def get_historical_data(symbol, timeframe, start_time, end_time):
    try:
        # Check if data is available in cache
        cache_key = f"{symbol}_{timeframe}_{start_time}_{end_time}"
        cached_data = requests_cache.get_cache().get(cache_key)
        if cached_data:
            return pd.read_json(cached_data)
        
        # Retrieve data from MT5
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        # ... Process and return data ...
        
        # Store data in cache
        requests_cache.get_cache().set(cache_key, data.to_json())
        
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