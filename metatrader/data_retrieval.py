import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from utils.error_handling import handle_error
from strategy.tunnel_strategy import get_data

def initialize_mt5():
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5")
        mt5.shutdown()
        return False
    return True

def shutdown_mt5():
    mt5.shutdown()

def get_historical_data(symbol, timeframe, start_time, end_time):
    """
    This function is replaced by get_data in the strategy.tunnel_strategy module.
    """
    return get_data(symbol, mode='backtest', start_date=start_time, end_date=end_time, timeframe=timeframe)

def get_current_price(symbol):
    """
    This function is replaced by get_data in the strategy.tunnel_strategy module.
    """
    return get_data(symbol, mode='live')

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

        # Use the new get_data function
        historical_data = get_data(symbol, mode='backtest', start_date=start_time, end_date=end_time, timeframe=timeframe)
        if historical_data is not None:
            print(f"Historical data for {symbol}:")
            print(historical_data.head())

        # Use the new get_data function for live price
        current_price = get_data(symbol, mode='live')
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

        orders = get_orders()  # Assign the result of the get_orders function to 'orders'
        if orders is not None:  # Check if 'orders' is not None
            print("Pending orders:")
            print(orders)


        shutdown_mt5()
