import numpy as np
from metatrader.data_retrieval import get_historical_data
from utils.error_handling import handle_error
from indicators import calculate_ema
from trade_management import execute_trade, manage_position

def calculate_tunnel_bounds(data, period, deviation):
    ema = calculate_ema(data['close'], period)
    upper_bound = ema + (deviation * np.std(data['close']))
    lower_bound = ema - (deviation * np.std(data['close']))
    return upper_bound, lower_bound

def generate_trade_signal(data, period, deviation):
    upper_bound, lower_bound = calculate_tunnel_bounds(data, period, deviation)
    
    if data['close'].iloc[-1] > upper_bound[-1]:
        return 'BUY'
    elif data['close'].iloc[-1] < lower_bound[-1]:
        return 'SELL'
    else:
        return None

def run_strategy(symbols, mt5_init, timeframe, lot_size, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    try:
        for symbol in symbols:
            data = get_historical_data(symbol, timeframe, start_time, end_time)
            if data is None:
                raise Exception(f"Failed to retrieve historical data for {symbol}")

            period = 20
            deviation = 2

            signal = generate_trade_signal(data, period, deviation)

            if signal == 'BUY':
                trade_request = {
                    'action': 'BUY',
                    'symbol': symbol,
                    'volume': lot_size,
                    'price': data['close'].iloc[-1],
                    'sl': data['close'].iloc[-1] - (1.5 * np.std(data['close'])),
                    'tp': data['close'].iloc[-1] + (2 * np.std(data['close'])),
                    'deviation': 10,
                    'magic': 12345,
                    'comment': 'Tunnel Strategy',
                    'type': 'ORDER_TYPE_BUY',
                    'type_filling': 'ORDER_FILLING_FOK',
                    'type_time': 'ORDER_TIME_GTC'
                }
                execute_trade(trade_request)
            elif signal == 'SELL':
                trade_request = {
                    'action': 'SELL',
                    'symbol': symbol,
                    'volume': lot_size,
                    'price': data['close'].iloc[-1],
                    'sl': data['close'].iloc[-1] + (1.5 * np.std(data['close'])),
                    'tp': data['close'].iloc[-1] - (2 * np.std(data['close'])),
                    'deviation': 10,
                    'magic': 12345,
                    'comment': 'Tunnel Strategy',
                    'type': 'ORDER_TYPE_SELL',
                    'type_filling': 'ORDER_FILLING_FOK',
                    'type_time': 'ORDER_TIME_GTC'
                }
                execute_trade(trade_request)

            manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

    except Exception as e:
        handle_error(e, "Failed to run the strategy")