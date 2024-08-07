from strategy.tunnel_strategy import execute_trade, manage_position
from utils.error_handling import handle_error
import logging

def calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value):
    risk_amount = balance * risk_percent
    if stop_loss_pips == 0 or pip_value == 0:
        logging.error("stop_loss_pips or pip_value cannot be zero.")
        return 0  # Return 0 or handle the error appropriately

    position_size = risk_amount / (stop_loss_pips * pip_value)
    return position_size

def entry_long(symbol, lot_size, price, sl, tp, deviation, magic, comment):
    trade_request = {
        'action': 'BUY',
        'symbol': symbol,
        'volume': lot_size,
        'price': price,
        'sl': sl,
        'tp': tp,
        'deviation': deviation,
        'magic': magic,
        'comment': comment,
        'type': 'ORDER_TYPE_BUY',
        'type_filling': 'ORDER_FILLING_FOK',
        'type_time': 'ORDER_TIME_GTC'
    }
    return execute_trade(trade_request)

def entry_short(symbol, lot_size, price, sl, tp, deviation, magic, comment):
    trade_request = {
        'action': 'SELL',
        'symbol': symbol,
        'volume': lot_size,
        'price': price,
        'sl': sl,
        'tp': tp,
        'deviation': deviation,
        'magic': magic,
        'comment': comment,
        'type': 'ORDER_TYPE_SELL',
        'type_filling': 'ORDER_FILLING_FOK',
        'type_time': 'ORDER_TIME_GTC'
    }
    return execute_trade(trade_request)

def exit_trade(position_ticket):
    try:
        close_request = {
            'action': 'CLOSE',
            'position': position_ticket,
            'type': 'ORDER_TYPE_CLOSE',
            'type_filling': 'ORDER_FILLING_FOK',
            'type_time': 'ORDER_TIME_GTC'
        }
        return execute_trade(close_request)
    except Exception as e:
        handle_error(e, "Failed to close the trade")
        return False
