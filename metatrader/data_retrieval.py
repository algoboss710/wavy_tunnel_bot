# metatrader/data_retrieval.py

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from utils.error_handling import handle_error
import logging

def initialize_mt5():
    if not mt5.initialize():
        logging.error("Failed to initialize MetaTrader5")
        mt5.shutdown()
        return False
    return True

def shutdown_mt5():
    mt5.shutdown()

def get_historical_data(symbol, timeframe, start_time, end_time):
    """
    This function fetches historical data from MT5.
    """
    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        logging.info(f"Fetched historical data for {symbol}: {len(rates)} candles")
        if rates is None or len(rates) == 0:
            raise ValueError(f"Failed to retrieve historical data for {symbol} with timeframe {timeframe} from {start_time} to {end_time}")

        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data = data.dropna()
        data = data[(data['open'] > 0) & (data['high'] > 0) & (data['low'] > 0) & (data['close'] > 0)]
        return data
    except Exception as e:
        handle_error(e, f"Failed to retrieve historical data for {symbol}")
        return None

def get_current_price(symbol):
    """
    This function gets the current price for a symbol.
    """
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"Failed to retrieve current price for {symbol}")
        return {'bid': tick.bid, 'ask': tick.ask, 'last': tick.last, 'time': datetime.now()}
    except Exception as e:
        handle_error(e, f"Failed to retrieve current price for {symbol}")
        return None

def get_account_info():
    account_info = mt5.account_info()
    if account_info:
        return account_info._asdict()
    else:
        logging.error("Failed to retrieve account information")
        return None

def get_available_symbols():
    symbols = mt5.symbols_get()
    if symbols:
        return [symbol.name for symbol in symbols]
    else:
        logging.error("Failed to retrieve available symbols")
        return None

def get_symbol_info(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        return symbol_info._asdict()
    else:
        logging.error(f"Failed to retrieve information for {symbol}")
        return None

def get_positions():
    positions = mt5.positions_get()
    if positions:
        return [position._asdict() for position in positions]
    else:
        logging.info("No open positions found")
        return []

def get_orders():
    orders = mt5.orders_get()
    if orders:
        return [order._asdict() for order in orders]
    else:
        logging.info("No pending orders found")
        return []

def get_data(symbol, mode='live', start_date=None, end_date=None, timeframe=mt5.TIMEFRAME_H1, num_candles=200):
    """
    Unified function to get either historical or live data.
    """
    try:
        if mode == 'live':
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=num_candles)
            historical_data = get_historical_data(symbol, timeframe, start_time, end_time)
            current_price = get_current_price(symbol)

            if historical_data is not None and current_price is not None:
                # Add current price to historical data
                current_data = pd.DataFrame([{
                    'time': current_price['time'],
                    'open': current_price['last'],
                    'high': current_price['last'],
                    'low': current_price['last'],
                    'close': current_price['last'],
                    'tick_volume': 0,
                    'spread': current_price['ask'] - current_price['bid'],
                    'real_volume': 0
                }])
                historical_data = pd.concat([historical_data, current_data]).reset_index(drop=True)

            return historical_data
        elif mode == 'backtest':
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date must be provided for backtest mode")
            return get_historical_data(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError("Invalid mode. Use 'live' or 'backtest'")
    except Exception as e:
        handle_error(e, f"Failed to get data for {symbol} in {mode} mode")
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if initialize_mt5():
        symbol = "EURUSD"
        timeframe = mt5.TIMEFRAME_H1

        live_data = get_data(symbol, mode='live', timeframe=timeframe)
        if live_data is not None:
            logging.info(f"Live data for {symbol}:")
            logging.info(live_data.tail())

        backtest_data = get_data(symbol, mode='backtest', start_date=datetime(2023, 1, 1), end_date=datetime.now(), timeframe=timeframe)
        if backtest_data is not None:
            logging.info(f"Backtest data for {symbol}:")
            logging.info(backtest_data.head())

        account_info = get_account_info()
        if account_info is not None:
            logging.info("Account information:")
            logging.info(account_info)

        available_symbols = get_available_symbols()
        if available_symbols is not None:
            logging.info("Available symbols:")
            logging.info(available_symbols[:5])  # Show first 5 symbols

        symbol_info = get_symbol_info(symbol)
        if symbol_info is not None:
            logging.info(f"Symbol information for {symbol}:")
            logging.info(symbol_info)

        positions = get_positions()
        if positions:
            logging.info("Open positions:")
            logging.info(positions)

        orders = get_orders()
        if orders:
            logging.info("Pending orders:")
            logging.info(orders)

        shutdown_mt5()