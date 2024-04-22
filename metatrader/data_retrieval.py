import MetaTrader5 as mt5

def get_historical_data(symbol, timeframe, start_time, end_time):
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    return rates

def get_current_price(symbol):
    prices = mt5.symbol_info_tick(symbol)
    return prices.last if prices else None

def get_account_info():
    return mt5.account_info()._asdict()
